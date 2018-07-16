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
from reikna.cluda import dtypes, functions, Snippet
import reikna.helpers as helpers
from reikna import transformations

from .computation_cache import get_computation
from .numeric_functions import Torus32, Float, dtot32_gpu, dtot32
from .random_numbers import rand_gaussian_float, rand_uniform_torus32, rand_gaussian_torus32


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
        #ks_b = Type(Torus32, (base, outer_n, t))
        #ks_cv = Type(Float, (base, outer_n, t))

        ks_a = Type(Torus32, (outer_n, t, base, inner_n))
        ks_b = Type(Torus32, (outer_n, t, base))
        ks_cv = Type(Float, (outer_n, t, base))

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
        """
        reduce_res = Type(ks_b.dtype, batch_shape + (outer_n, t))

        l = len(reduce_res.shape)
        self._sum_ks_b = Reduce(reduce_res, predicate_sum(ks_b.dtype), axes=(l-2, l-1))

        trf = filter_by_aijs_trf_single(aijs, ks_b)
        sub_trf = sub_from(b)

        self._sum_ks_b.parameter.input.connect(
            trf, trf.output, ks_b=trf.full_ks_b, aijs=trf.aijs)

        self._sum_ks_b.parameter.output.connect(
            sub_trf, sub_trf.to_sub, b=sub_trf.input, result_b=sub_trf.output)
        """
        # cv
        """
        reduce_res = Type(ks_cv.dtype, batch_shape + (outer_n, t))

        l = len(reduce_res.shape)
        self._sum_ks_cv = Reduce(reduce_res, predicate_sum(ks_cv.dtype), axes=(l-2, l-1))

        trf = filter_by_aijs_trf_single(aijs, ks_cv)
        add_trf = add_to(cv)

        self._sum_ks_cv.parameter.input.connect(
            trf, trf.output, ks_cv=trf.full_ks_b, aijs=trf.aijs)

        self._sum_ks_cv.parameter.output.connect(
            add_trf, add_trf.to_add, cv=add_trf.input, result_cv=add_trf.output)
        """

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

        #aijs = plan.temp_array_like(self._prepare_aijs.parameter.aijs)

        #plan.computation_call(self._prepare_aijs, aijs, ai)

        #plan.computation_call(self._sum_ks_a, result_a, result_a, aijs, ks_a)

        # result_a: batch_shape + (inner_n,)
        # ks_a: (inner_n, base, outer_n, t)
        # ai: batch_shape + (outer_n,)
        batch_shape = result_a.shape[:-1]
        plan.kernel_call(
            TEMPLATE.get_def("keyswitch"),
            [result_a, result_b, result_cv, ks_a, ks_b, ks_cv, ai],
            global_size=(helpers.product(batch_shape), 512),
            local_size=(1, 512),
            render_kwds=dict(
                slices=(len(batch_shape), 1)
                )
            )

        #plan.computation_call(self._sum_ks_b, result_b, result_b, aijs, ks_b)
        #plan.computation_call(self._sum_ks_cv, result_cv, result_cv, aijs, ks_cv)

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


class MatrixMulVector(Computation):

    def __init__(self, arr):
        Computation.__init__(self,
            [Parameter('output', Annotation(Type(arr.dtype, arr.shape[:-1]), 'o')),
            Parameter('matrix', Annotation(arr, 'i')),
            Parameter('vector', Annotation(Type(arr.dtype, arr.shape[-1]), 'i'))
            ])

    def _build_plan(self, plan_factory, device_params, output, matrix, vector):
        plan = plan_factory()

        summation = Reduce(matrix, predicate_sum(matrix.dtype), axes=(len(matrix.shape)-1,))

        mul_vec = Transformation(
        [
            Parameter('output', Annotation(matrix, 'o')),
            Parameter('matrix', Annotation(matrix, 'i')),
            Parameter('vector', Annotation(vector, 'i')),
        ],
        """
        ${output.store_same}(${mul}(${matrix.load_same}, ${vector.load_idx}(${idxs[-1]})));
        """,
        render_kwds=dict(mul=functions.mul(matrix.dtype, vector.dtype)),
        connectors=['output', 'matrix'])

        summation.parameter.input.connect(
            mul_vec, mul_vec.output, matrix=mul_vec.matrix, vector=mul_vec.vector)

        plan.computation_call(summation, output, matrix, vector)

        return plan


class LweKeySwitchKeyComputation(Computation):

    def __init__(self, extracted_n: int, t: int, basebit: int, inner_n: int, alpha):

        base = 1 << basebit

        a = Type(numpy.int32, (extracted_n, t, base, inner_n))
        b = Type(numpy.int32, (extracted_n, t, base))
        cv = Type(numpy.float64, (extracted_n, t, base))
        in_key = Type(numpy.int32, extracted_n)
        out_key = Type(numpy.int32, inner_n)

        noises_a = Type(numpy.int32, (extracted_n, t, base - 1, inner_n))
        noises_b = Type(numpy.float64, (extracted_n, t, base - 1))

        self._basebit = basebit
        self._alpha = alpha

        Computation.__init__(self,
            [Parameter('ks_a', Annotation(a, 'o')),
            Parameter('ks_b', Annotation(b, 'o')),
            Parameter('ks_cv', Annotation(cv, 'o')),
            Parameter('in_key', Annotation(in_key, 'i')),
            Parameter('out_key', Annotation(out_key, 'i')),
            Parameter('noises_a', Annotation(noises_a, 'i')),
            Parameter('noises_b', Annotation(noises_b, 'i')),
            ])

    def _build_plan(
            self, plan_factory, device_params,
            ks_a, ks_b, ks_cv, in_key, out_key, noises_a, noises_b):

        plan = plan_factory()

        extracted_n, t, base, inner_n = ks_a.shape

        mean = Reduce(noises_b, predicate_sum(noises_b.dtype))
        norm = transformations.div_const(mean.parameter.output, numpy.prod(noises_b.shape))
        mean.parameter.output.connect(norm, norm.input, mean=norm.output)

        noises_b_mean = plan.temp_array_like(mean.parameter.mean)

        mul_key = MatrixMulVector(noises_a)
        b_term = plan.temp_array_like(mul_key.parameter.output)

        plan.computation_call(mean, noises_b_mean, noises_b)
        plan.computation_call(mul_key, b_term, noises_a, out_key)

        build_keyswitch = PureParallel(
            [
                Parameter('ks_a', Annotation(ks_a, 'o')),
                Parameter('ks_b', Annotation(ks_b, 'o')),
                Parameter('ks_cv', Annotation(ks_cv, 'o')),
                Parameter('in_key', Annotation(in_key, 'i')),
                Parameter('b_term', Annotation(b_term, 'i')),
                Parameter('noises_a', Annotation(noises_a, 'i')),
                Parameter('noises_b', Annotation(noises_b, 'i')),
                Parameter('noises_b_mean', Annotation(noises_b_mean, 'i'))],
            Snippet(
                TEMPLATE.get_def("build_keyswitch_key"),
                render_kwds=dict(
                    extracted_n=extracted_n, t=t, basebit=self._basebit, inner_n=inner_n,
                    dtot32=dtot32_gpu, alpha=self._alpha)),
            guiding_array="ks_b")

        plan.computation_call(
            build_keyswitch,
            ks_a, ks_b, ks_cv, in_key, b_term, noises_a, noises_b, noises_b_mean
            )

        return plan


def vec_mul_mat(b, a):
    return (a * b).sum(-1, dtype=numpy.int32)


# This function encrypts a message by using key and a given noise value
def lweSymEncryptWithExternalNoise(
        ks_a, ks_b, ks_cv, messages, a_noises, b_noises, alpha: float, key):

    # term h=0 as trivial encryption of 0 (it will not be used in the KeySwitching)
    ks_a[:,:,0,:] = 0
    ks_b[:,:,0] = 0
    ks_cv[:,:,0] = 0

    ks_b[:,:,1:] = messages + dtot32(b_noises)
    ks_a[:,:,1:,:] = a_noises
    ks_b[:,:,1:] += vec_mul_mat(key, a_noises)
    ks_cv[:,:,1:] = alpha**2


def LweKeySwitchKeyComputation_ref(extracted_n: int, t: int, basebit: int, inner_n: int, alpha):

    base = 1 << basebit

    def _kernel(ks_a, ks_b, ks_cv, in_key, out_key, a_noises, b_noises):

        # recenter the noises
        b_noises -= b_noises.mean()

        # generate the ks

        # mess::Torus32 = (in_key.key[i] * Int32(h - 1)) * Int32(1 << (32 - j * basebit))
        hs = numpy.arange(2, base+1)
        js = numpy.arange(1, t+1)

        r_key = in_key.reshape(extracted_n, 1, 1)
        r_hs = hs.reshape(1, 1, base - 1)
        r_js = js.reshape(1, t, 1)

        messages = r_key * (r_hs - 1) * (1 << (32 - r_js * basebit))
        messages = messages.astype(Torus32)

        lweSymEncryptWithExternalNoise(ks_a, ks_b, ks_cv, messages, a_noises, b_noises, alpha, out_key)

    return _kernel


def LweKeySwitchKey_gpu(
        thr, rng, ks, extracted_n: int, t: int, basebit: int, in_key: 'LweKey', out_key: 'LweKey'):

    inner_n = out_key.params.n
    alpha = out_key.params.alpha_min

    comp = get_computation(
        thr, LweKeySwitchKeyComputation,
        extracted_n, t, basebit, inner_n, alpha
        )

    base = 1 << basebit
    b_noises = rand_gaussian_float(thr, rng, alpha, (extracted_n, t, base - 1))
    a_noises = rand_uniform_torus32(thr, rng, (extracted_n, t, base - 1, inner_n))
    comp(
        ks.a, ks.b, ks.current_variances,
        in_key.key, out_key.key, a_noises, b_noises)


# * This function encrypts message by using key, with stdev alpha
# * The Lwe sample for the result must be allocated and initialized
# * (this means that the parameters are already in the result)
def LweSymEncrypt_ref(shape, n, alpha: float):

    def _kernel(result_a, result_b, result_cv, messages, key, noises_a, noises_b):
        numpy.copyto(result_b, noises_b + messages)
        numpy.copyto(result_a, noises_a)
        result_b += vec_mul_mat(key, result_a)
        result_cv.fill(alpha**2)

    return _kernel


class LweSymEncrypt(Computation):

    def __init__(self, shape, n, alpha):

        a = Type(numpy.int32, shape + (n,))
        b = Type(numpy.int32, shape)
        cv = Type(numpy.float64, shape)
        messages = Type(numpy.int32, shape)
        key = Type(numpy.int32, (n,))
        noises_a = Type(Torus32, shape + (n,))
        noises_b = Type(Torus32, shape)

        self._alpha = alpha

        Computation.__init__(self,
            [Parameter('result_a', Annotation(a, 'o')),
            Parameter('result_b', Annotation(b, 'o')),
            Parameter('result_cv', Annotation(cv, 'o')),
            Parameter('messages', Annotation(messages, 'i')),
            Parameter('key', Annotation(key, 'i')),
            Parameter('noises_a', Annotation(noises_a, 'i')),
            Parameter('noises_b', Annotation(noises_b, 'i')),
            ])

    def _build_plan(
            self, plan_factory, device_params,
            result_a, result_b, result_cv, messages, key, noises_a, noises_b):

        plan = plan_factory()

        mul_key = MatrixMulVector(noises_a)

        fill_b_cv = Transformation(
            [
                Parameter('result_b', Annotation(result_b, 'o')),
                Parameter('result_cv', Annotation(result_cv, 'o')),
                Parameter('messages', Annotation(messages, 'i')),
                Parameter('noises_a_times_key', Annotation(noises_b, 'i')),
                Parameter('noises_b', Annotation(noises_b, 'i')),
            ],
            """
            ${result_b.store_same}(
                ${noises_b.load_same}
                + ${messages.load_same}
                + ${noises_a_times_key.load_same});
            ${result_cv.store_same}(${alpha**2});
            """,
            connectors=['noises_a_times_key'],
            render_kwds=dict(alpha=self._alpha))

        mul_key.parameter.output.connect(
            fill_b_cv, fill_b_cv.noises_a_times_key,
            b=fill_b_cv.result_b, cv=fill_b_cv.result_cv, messages=fill_b_cv.messages,
            noises_b=fill_b_cv.noises_b)

        plan.computation_call(mul_key, result_b, result_cv, messages, noises_b, noises_a, key)
        plan.computation_call(
            PureParallel.from_trf(transformations.copy(noises_a)),
            result_a, noises_a)

        return plan


def lweSymEncrypt_gpu(thr, rng, result: 'LweSampleArray', messages, alpha: float, key: 'LweKey'):
    n = key.params.n
    noises_b = rand_gaussian_torus32(thr, rng, 0, alpha, messages.shape)
    noises_a = rand_uniform_torus32(thr, rng, messages.shape + (n,))
    comp = get_computation(thr, LweSymEncrypt, messages.shape, n, alpha)
    comp(result.a, result.b, result.current_variances, messages, key.key, noises_a, noises_b)


# This function computes the phase of sample by using key : phi = b - a.s
def LwePhase_ref(shape, n):

    def _kernel(result, a, b, key):
        numpy.copyto(result, b - vec_mul_mat(key, a))

    return _kernel


class LwePhase(Computation):

    def __init__(self, shape, n):

        a = Type(numpy.int32, shape + (n,))
        b = Type(numpy.int32, shape)
        key = Type(numpy.int32, (n,))

        Computation.__init__(self,
            [Parameter('result', Annotation(b, 'o')),
            Parameter('a', Annotation(a, 'i')),
            Parameter('b', Annotation(b, 'i')),
            Parameter('key', Annotation(key, 'i')),
            ])

    def _build_plan(
            self, plan_factory, device_params, result, a, b, key):

        plan = plan_factory()

        mul_key = MatrixMulVector(a)

        fill_res = Transformation(
            [
                Parameter('result', Annotation(result, 'o')),
                Parameter('b', Annotation(b, 'i')),
                Parameter('a_times_key', Annotation(b, 'i')),
            ],
            """
            ${result.store_same}(${b.load_same} - ${a_times_key.load_same});
            """,
            connectors=['a_times_key'])

        mul_key.parameter.output.connect(
            fill_res, fill_res.a_times_key,
            result=fill_res.result, b=fill_res.b)

        plan.computation_call(mul_key, result, b, a, key)

        return plan


def lwePhase_gpu(thr, sample: 'LweSampleArray', key: 'LweKey'):
    comp = get_computation(thr, LwePhase, sample.shape, key.params.n)
    result = thr.empty_like(sample.b)
    comp(result, sample.a, sample.b, key.key)
    return result.get()
