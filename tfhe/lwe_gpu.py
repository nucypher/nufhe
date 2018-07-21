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
from reikna.cluda import functions, Snippet
import reikna.helpers as helpers
from reikna import transformations

from .computation_cache import get_computation
from .numeric_functions import Torus32, Float, dtot32_gpu
from .random_numbers import rand_gaussian_float, rand_uniform_torus32, rand_gaussian_torus32


TEMPLATE = helpers.template_for(__file__)


class LweKeySwitchTranslate_fromArray(Computation):

    def __init__(self, batch_shape, t: int, outer_n, inner_n, basebit: int):

        base = 1 << basebit

        a = Type(Torus32, batch_shape + (inner_n,))
        b = Type(Torus32, batch_shape)
        cv = Type(Float, batch_shape)

        ks_a = Type(Torus32, (outer_n, t, base, inner_n))
        ks_b = Type(Torus32, (outer_n, t, base))
        ks_cv = Type(Float, (outer_n, t, base))

        ai = Type(Torus32, batch_shape + (outer_n,))

        self._t = t
        self._outer_n = outer_n
        self._inner_n = inner_n
        self._basebit = basebit

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

        batch_shape = result_a.shape[:-1]
        plan.kernel_call(
            TEMPLATE.get_def("keyswitch"),
            [result_a, result_b, result_cv, ks_a, ks_b, ks_cv, ai],
            global_size=(helpers.product(batch_shape), 512),
            local_size=(1, 512),
            render_kwds=dict(
                slices=(len(batch_shape), 1),
                lwe_n=self._inner_n,
                tlwe_n=self._outer_n,
                decomp_bits=self._basebit,
                decomp_size=self._t,
                )
            )

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


def LweKeySwitchKey_gpu(
        thr, rng, ks, extracted_n: int, t: int, basebit: int, in_key: 'LweKey', out_key: 'LweKey'):

    inner_n = out_key.params.size
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
    n = key.params.size
    noises_b = rand_gaussian_torus32(thr, rng, 0, alpha, messages.shape)
    noises_a = rand_uniform_torus32(thr, rng, messages.shape + (n,))
    comp = get_computation(thr, LweSymEncrypt, messages.shape, n, alpha)
    comp(result.a, result.b, result.current_variances, messages, key.key, noises_a, noises_b)


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
    comp = get_computation(thr, LwePhase, sample.shape, key.params.size)
    result = thr.empty_like(sample.b)
    comp(result, sample.a, sample.b, key.key)
    return result.get()
