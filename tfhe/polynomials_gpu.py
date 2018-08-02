import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation
from reikna.algorithms import PureParallel
from reikna.cluda import dtypes


def transform_mul_by_xai(ais, arr, ai_view=False, minus_one=False, invert_ais=False):
    # arr: ... x N
    # ais: ..., int

    if ai_view:
        assert len(ais.shape) == len(arr.shape) - 1
        assert ais.shape[:-1] == arr.shape[:-2]
    else:
        assert len(ais.shape) == len(arr.shape) - 1
        assert ais.shape == arr.shape[:-1]

    N = arr.shape[-1]

    return Transformation(
        [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('ais', Annotation(ais, 'i')),
            Parameter('ai_idx', Annotation(numpy.int32)), # FIXME: unused if ai_view==False
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        %if ai_view:
        ${ai_ctype} ai = ${ais.load_idx}(${", ".join(idxs[:batch_len])}, ${ai_idx});
        %else:
        ${ai_ctype} ai = ${ais.load_idx}(${", ".join(idxs[:batch_len])});
        %endif

        %if invert_ais:
        ai = ${2 * N} - ai;
        %endif

        ${output.ctype} res;

        if (ai < ${N})
        {
            if (${idxs[-1]} < ai)
            {
                res = -${input.load_idx}(
                        ${", ".join(idxs[:-1])}, ${idxs[-1]} + ${N} - ai
                        );
            }
            else
            {
                res = ${input.load_idx}(
                        ${", ".join(idxs[:-1])}, ${idxs[-1]} - ai
                        );
            }
        }
        else
        {
            ${ai_ctype} aa = ai - ${N};
            if (${idxs[-1]} < aa)
            {
                res = ${input.load_idx}(
                        ${", ".join(idxs[:-1])}, ${idxs[-1]} + ${N} - aa
                        );
            }
            else
            {
                res = -${input.load_idx}(
                      ${", ".join(idxs[:-1])}, ${idxs[-1]} - aa
                      );
            }
        }

        %if minus_one:
        res -= ${input.load_same};
        %endif

        ${output.store_same}(res);
        """,
        render_kwds=dict(
            batch_len=len(arr.shape) - 1 - int(ai_view),
            N=N, ai_ctype=dtypes.ctype(ais.dtype),
            ai_view=ai_view, minus_one=minus_one, invert_ais=invert_ais),
        connectors=['output'])


class ShiftTorusPolynomial(Computation):

    def __init__(self, ais, arr, ai_view=False, minus_one=False, invert_ais=False):
        # `invert_ais` means that `2N - ais` will be used instead of `ais`
        tr = transform_mul_by_xai(
            ais, arr, ai_view=ai_view, minus_one=minus_one, invert_ais=invert_ais)
        self._pp = PureParallel.from_trf(tr, guiding_array=tr.output)

        Computation.__init__(self, [
            Parameter('output', Annotation(self._pp.parameter.output, 'o')),
            Parameter('ais', Annotation(self._pp.parameter.ais, 'i')),
            Parameter('ai_idx', Annotation(numpy.int32)), # FIXME: unused if ai_view==False
            Parameter('input', Annotation(self._pp.parameter.input, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, ais, ai_idx, input_):
        plan = plan_factory()
        plan.computation_call(self._pp, output, ais, ai_idx, input_)
        return plan
