"""
 * translates the message of the result sample by -sum(a[i].s[i]) where s is the secret
 * embedded in ks.
 * @param result the LWE sample to translate by -sum(ai.si).
 * @param ks The (n x t x base) key switching key
 *        ks[i][j][k] encodes k.s[i]/base^(j+1)
 * @param params The common LWE parameters of ks and result
 * @param ai The input torus array
 * @param n The size of the input key
 * @param t The precision of the keyswitch (technically, 1/2.base^t)
 * @param basebit Log_2 of base
"""

import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.algorithms import PureParallel, Reduce, predicate_sum
from reikna.cluda import dtypes, functions
import reikna.helpers as helpers

from .computation_cache import get_computation
from .numeric_functions import Torus32, Float


TEMPLATE = helpers.template_for(__file__)


def prepare_aijs_trf(ai, t, basebit):
    aijs = Type(ai.dtype, ai.shape + (t,))

    base = 1 << basebit # base=2 in [CGGI16]
    prec_offset = 1 << (32 - (1 + basebit * t)) # precision
    mask = base - 1

    return Transformation(
        [
            Parameter('aijs', Annotation(aijs, 'o')),
            Parameter('ai', Annotation(ai, 'i'))
        ],
        """
        ${aijs.store_same}(
            ((${ai.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-2]}) + (${prec_offset}))
                >> (32 - (${idxs[-1]} + 1) * ${basebit})) & ${mask}
            );
        """,
        render_kwds=dict(basebit=basebit, mask=mask, prec_offset=prec_offset))


def filter_by_aijs_trf(aijs, ks_a_tr):
    batch_shape = aijs.shape[:-2]
    inner_n, base, outer_n, t = ks_a_tr.shape
    filtered_ks = Type(ks_a_tr.dtype, batch_shape + (inner_n, outer_n, t))

    return Transformation(
        [
            Parameter('output', Annotation(filtered_ks, 'o')),
            Parameter('aijs', Annotation(aijs, 'i')),
            Parameter('full_ks_a', Annotation(ks_a_tr, 'i'))
        ],
        """
        ${aijs.ctype} x = ${aijs.load_idx}(${", ".join(idxs[:-3])}, ${idxs[-2]}, ${idxs[-1]});
        ${output.ctype} res;
        if (x != 0)
        {
            res = ${full_ks_a.load_idx}(${idxs[-3]}, x, ${idxs[-2]}, ${idxs[-1]});
        }
        else
        {
            res = 0;
        }
        ${output.store_same}(res);
        """,
        connectors=['output'])


def filter_by_aijs_trf_single(aijs, ks_b):
    batch_shape = aijs.shape[:-2]
    base, outer_n, t = ks_b.shape
    filtered_ks = Type(ks_b.dtype, batch_shape + (outer_n, t))

    return Transformation(
        [
            Parameter('output', Annotation(filtered_ks, 'o')),
            Parameter('aijs', Annotation(aijs, 'i')),
            Parameter('full_ks_b', Annotation(ks_b, 'i'))
        ],
        """
        ${aijs.ctype} x = ${aijs.load_idx}(${", ".join(idxs)});
        ${output.ctype} res;
        if (x != 0)
        {
            res = ${full_ks_b.load_idx}(x, ${idxs[-2]}, ${idxs[-1]});
        }
        else
        {
            res = 0;
        }
        ${output.store_same}(res);
        """,
        connectors=['output'])



def sub_from(a):
    return Transformation(
        [
            Parameter('output', Annotation(a, 'o')),
            Parameter('input', Annotation(a, 'i')),
            Parameter('to_sub', Annotation(a, 'i')),
        ],
        """
        ${output.store_same}(${input.load_same} - ${to_sub.load_same});
        """,
        connectors=['output'])


def add_to(a):
    return Transformation(
        [
            Parameter('output', Annotation(a, 'o')),
            Parameter('input', Annotation(a, 'i')),
            Parameter('to_add', Annotation(a, 'i')),
        ],
        """
        ${output.store_same}(${input.load_same} + ${to_add.load_same});
        """,
        connectors=['output'])


class LweKeySwitchTranslate_fromArray(Computation):

    def __init__(self, batch_shape, t: int, outer_n, inner_n, basebit: int):

        base = 1 << basebit # base=2 in [CGGI16]

        a = Type(Torus32, batch_shape + (inner_n,))
        b = Type(Torus32, batch_shape)
        cv = Type(Float, batch_shape)
        #ks_a = Type(Torus32, (inner_n, base, outer_n, t))
        ks_a = Type(Torus32, (outer_n, t, base, inner_n))
        ks_b = Type(Torus32, (base, outer_n, t))
        ks_cv = Type(Float, (base, outer_n, t))
        ai = Type(Torus32, batch_shape + (outer_n,))

        self._prepare_aijs = PureParallel.from_trf(prepare_aijs_trf(ai, t, basebit))
        aijs = self._prepare_aijs.parameter.aijs

        # a
        """
        reduce_res = Type(ks_a.dtype, batch_shape + (inner_n, outer_n, t))

        l = len(reduce_res.shape)
        self._sum_ks_a = Reduce(reduce_res, predicate_sum(ks_a.dtype), axes=(l-2, l-1))

        trf = filter_by_aijs_trf(aijs, ks_a)
        sub_trf = sub_from(a)

        self._sum_ks_a.parameter.input.connect(
            trf, trf.output, ks_a=trf.full_ks_a, aijs=trf.aijs)

        self._sum_ks_a.parameter.output.connect(
            sub_trf, sub_trf.to_sub, a=sub_trf.input, result_a=sub_trf.output)
        """
        # b

        reduce_res = Type(ks_b.dtype, batch_shape + (outer_n, t))

        l = len(reduce_res.shape)
        self._sum_ks_b = Reduce(reduce_res, predicate_sum(ks_b.dtype), axes=(l-2, l-1))

        trf = filter_by_aijs_trf_single(aijs, ks_b)
        sub_trf = sub_from(b)

        self._sum_ks_b.parameter.input.connect(
            trf, trf.output, ks_b=trf.full_ks_b, aijs=trf.aijs)

        self._sum_ks_b.parameter.output.connect(
            sub_trf, sub_trf.to_sub, b=sub_trf.input, result_b=sub_trf.output)

        # cv

        reduce_res = Type(ks_cv.dtype, batch_shape + (outer_n, t))

        l = len(reduce_res.shape)
        self._sum_ks_cv = Reduce(reduce_res, predicate_sum(ks_cv.dtype), axes=(l-2, l-1))

        trf = filter_by_aijs_trf_single(aijs, ks_cv)
        add_trf = add_to(cv)

        self._sum_ks_cv.parameter.input.connect(
            trf, trf.output, ks_cv=trf.full_ks_b, aijs=trf.aijs)

        self._sum_ks_cv.parameter.output.connect(
            add_trf, add_trf.to_add, cv=add_trf.input, result_cv=add_trf.output)


        Computation.__init__(self,
            [Parameter('result_a', Annotation(a, 'io')),
            Parameter('result_b', Annotation(b, 'io')),
            Parameter('result_cv', Annotation(cv, 'io')),
            Parameter('ks_a', Annotation(ks_a, 'i')),
            Parameter('ks_b', Annotation(ks_b, 'i')),
            Parameter('ks_cv', Annotation(ks_cv, 'i')),
            Parameter('ai', Annotation(ai, 'i'))])

    def _build_plan(
            self, plan_factory, device_params,
            result_a, result_b, result_cv,
            ks_a, ks_b, ks_cv, ai):

        plan = plan_factory()

        aijs = plan.temp_array_like(self._prepare_aijs.parameter.aijs)

        plan.computation_call(self._prepare_aijs, aijs, ai)

        #plan.computation_call(self._sum_ks_a, result_a, result_a, aijs, ks_a)

        # result_a: batch_shape + (inner_n,)
        # ks_a: (inner_n, base, outer_n, t)
        # ai: batch_shape + (outer_n,)
        batch_shape = result_a.shape[:-1]
        plan.kernel_call(
            TEMPLATE.get_def("keyswitch"),
            [result_a, ks_a, ai],
            global_size=(helpers.product(batch_shape), 512),
            local_size=(1, 512),
            render_kwds=dict(
                slices=(len(batch_shape), 1)
                )
            )

        plan.computation_call(self._sum_ks_b, result_b, result_b, aijs, ks_b)
        plan.computation_call(self._sum_ks_cv, result_cv, result_cv, aijs, ks_cv)

        return plan


def lweKeySwitchTranslate_fromArray_gpu(result, ks, params, a, outer_n, t, basebit):

    batch_shape = result.a.shape[:-1]
    inner_n = result.a.shape[-1]
    thr = result.a.thread

    comp = get_computation(
        thr, LweKeySwitchTranslate_fromArray,
        batch_shape, t, outer_n, inner_n, basebit)
    comp(
        result.a, result.b, result.current_variances,
        ks.a, ks.b, ks.current_variances,
        a)
