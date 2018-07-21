import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation
from reikna.algorithms import PureParallel
from reikna.cluda import dtypes

from .computation_cache import get_computation
from .numeric_functions import Torus32
from .polynomial_transform import transformed_length, transformed_dtype


class Polynomial:

    def flat_coefs(self):
        cp = _coefs(self)
        return cp.reshape(numpy.prod(self.shape), cp.shape[-1])

    def polynomial_size(self):
        return self._polynomial_size


class FakeThread:

    def array(self, shape, dtype):
        return numpy.empty(shape, dtype)


# This structure represents an integer polynomial modulo X^N+1
class IntPolynomialArray(Polynomial):
    def __init__(self, thr, N, shape):
        Polynomial.__init__(self)

        if thr is None:
            thr = FakeThread()

        self.coefs = thr.array(shape + (N,), numpy.int32)
        self._polynomial_size = N
        self.shape = shape

    @classmethod
    def from_array(cls, arr):
        obj = cls(arr.thread, arr.shape[-1], arr.shape[:-1])
        obj.coefs = arr
        return obj


# This structure represents an torus polynomial modulo X^N+1
class TorusPolynomialArray(Polynomial):
    def __init__(self, thr, N, shape):
        Polynomial.__init__(self)

        if thr is None:
            thr = FakeThread()

        self.coefsT = thr.array(shape + (N,), Torus32)
        self._polynomial_size = N
        self.shape = shape

    @classmethod
    def from_arr(cls, arr):
        obj = cls(arr.shape[-1], arr.shape[:-1])
        obj.coefsT = arr
        return obj


# This structure is used for FFT operations, and is a representation
# over C of a polynomial in R[X]/X^N+1
class LagrangeHalfCPolynomialArray(Polynomial):
    def __init__(self, thr, N, shape):
        Polynomial.__init__(self)
        assert N % 2 == 0

        if thr is None:
            thr = FakeThread()

        self.coefsC = thr.array(shape + (transformed_length(N),), transformed_dtype())
        self._polynomial_size = N
        self.shape = shape


def _coefs(p):
    # TODO: different field names help with debugging, remove later
    if type(p) == IntPolynomialArray:
        return p.coefs
    elif type(p) == TorusPolynomialArray:
        return p.coefsT
    elif type(p) == LagrangeHalfCPolynomialArray:
        return p.coefsC
    else:
        raise Exception("Wrong type " + str(type(p)))



def transform_mul_by_xai(ais, arr, ai_view=False, minus_one=False, invert_ais=False):
    # arr: ... x N
    # ais: arr.shape[0], int

    assert len(ais.shape) == 2 if ai_view else 1
    assert ais.shape[0] == arr.shape[0]
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
        ${ai_ctype} ai = ${ais.load_idx}(${idxs[0]}, ${ai_idx});
        %else:
        ${ai_ctype} ai = ${ais.load_idx}(${idxs[0]});
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
            N=N, ai_ctype=dtypes.ctype(ais.dtype),
            ai_view=ai_view, minus_one=minus_one, invert_ais=invert_ais),
        connectors=['output'])


class TPMulByXai(Computation):

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


# result= X^{a}*source
def tp_mul_by_xai_gpu(out: TorusPolynomialArray, ais, in_: TorusPolynomialArray, invert_ais=False):
    thr = out.coefsT.thread
    comp = get_computation(
        thr, TPMulByXai, ais, in_.coefsT, minus_one=False, invert_ais=invert_ais)
    comp(out.coefsT, ais, 0, in_.coefsT)


# result = (X^ai-1) * source
def tp_mul_by_xai_minus_one_gpu(out: TorusPolynomialArray, ais, ai_idx, in_: TorusPolynomialArray):
    thr = out.coefsT.thread
    comp = get_computation(
        thr, TPMulByXai, ais, in_.coefsT, ai_view=True, minus_one=True, invert_ais=False)
    comp(out.coefsT, ais, ai_idx, in_.coefsT)
