import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.algorithms import PureParallel
import reikna.helpers as helpers

from .tlwe import TLweSampleArray, TLweParams
from .lwe import LweSampleArray, LweParams
from .computation_cache import get_computation
from .gpu_polynomials import TorusPolynomialArray, tp_mul_by_xai_minus_one_gpu
from .numeric_functions import Torus32, Float
from .random_numbers import rand_gaussian_torus32, rand_uniform_torus32
from .polynomial_transform import get_transform
from .performance import PerformanceParameters, performance_parameters_for_device


TEMPLATE = helpers.template_for(__file__)


class TLweNoiselessTrivial(Computation):

    def __init__(self, a):
        cv_type = Type(Float, a.shape[:-2])
        mu_type = Type(a.dtype, a.shape[:-2] + (a.shape[-1],))

        self._fill_a = PureParallel(
            [Parameter('a', Annotation(a, 'o')),
            Parameter('mu', Annotation(mu_type, 'i'))],
            """
            ${a.ctype} a;
            if (${idxs[-2]} == ${k})
            {
                a = ${mu.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-1]});
            }
            else
            {
                a = 0;
            }
            ${a.store_same}(a);
            """,
            render_kwds=dict(k=a.shape[-2] - 1))

        self._fill_cv = PureParallel(
            [Parameter('current_variances', Annotation(cv_type, 'o'))],
            """
            ${current_variances.store_same}(0);
            """)

        Computation.__init__(self,
            [Parameter('a', Annotation(a, 'o')),
            Parameter('current_variances', Annotation(cv_type, 'o')),
            Parameter('mu', Annotation(mu_type, 'i'))])

    def _build_plan(self, plan_factory, device_params, a, current_variances, mu):
        plan = plan_factory()
        plan.computation_call(self._fill_a, a, mu)
        plan.computation_call(self._fill_cv, current_variances)
        return plan


# result = (0,mu)
def tLweNoiselessTrivial_gpu(result: TLweSampleArray, mu: TorusPolynomialArray, params: TLweParams):
    thr = result.a.coefsT.thread
    comp = get_computation(thr, TLweNoiselessTrivial, result.a.coefsT)
    comp(result.a.coefsT, result.current_variances, mu.coefsT)


# TODO: can be made faster by using local memory
class TLweExtractLweSample(Computation):

    def __init__(self, tlwe_a):

        assert len(tlwe_a.shape) > 2
        k = tlwe_a.shape[-2] - 1
        N = tlwe_a.shape[-1]

        batch_shape = tlwe_a.shape[:-2]
        a = Type(tlwe_a.dtype, batch_shape + (k * N,))
        b = Type(tlwe_a.dtype, batch_shape)

        self._fill_a = PureParallel(
            [Parameter('a', Annotation(a, 'o')),
            Parameter('tlwe_a', Annotation(tlwe_a, 'i'))],
            """
            ${a.ctype} a;
            if (${idxs[-1]} == 0)
            {
                a = ${tlwe_a.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-2]}, 0);
            }
            else
            {
                a = -${tlwe_a.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-2]}, ${N} - ${idxs[-1]});
            }
            ${a.store_idx}(${", ".join(idxs[:-2])}, ${idxs[-2]} * ${N} + ${idxs[-1]}, a);
            """,
            render_kwds=dict(k=k, N=N),
            guiding_array=batch_shape + (k, N))

        self._fill_b = PureParallel(
            [Parameter('b', Annotation(b, 'o')),
            Parameter('tlwe_a', Annotation(tlwe_a, 'i'))],
            """
            ${b.store_same}(${tlwe_a.load_idx}(${", ".join(idxs)}, ${k}, 0));
            """,
            render_kwds=dict(k=k))

        Computation.__init__(self,
            [Parameter('a', Annotation(a, 'o')),
            Parameter('b', Annotation(b, 'o')),
            Parameter('tlwe_a', Annotation(tlwe_a, 'i'))])

    def _build_plan(self, plan_factory, device_params, a, b, tlwe_a):
        plan = plan_factory()
        plan.computation_call(self._fill_a, a, tlwe_a)
        plan.computation_call(self._fill_b, b, tlwe_a)
        return plan


def tLweExtractLweSample_gpu(
        result: LweSampleArray, x: TLweSampleArray, params: LweParams, rparams: TLweParams):
    thr = result.a.thread
    comp = get_computation(thr, TLweExtractLweSample, x.a.coefsT)
    comp(result.a, result.b, x.a.coefsT)


# mult externe de X^ai-1 par bki
def tLweMulByXaiMinusOne_gpu(result:TLweSampleArray, ai, ai_idx, bk: TLweSampleArray, params: TLweParams):
    # TYPING: ai::Array{Int32}
    tp_mul_by_xai_minus_one_gpu(result.a, ai, ai_idx, bk.a)


# result = result + sample
def tLweAddTo_gpu(result: TLweSampleArray, accum: TLweSampleArray, params: TLweParams):
    result.a.coefsT += accum.a.coefsT
    result.current_variances += accum.current_variances


# result = sample
def tLweCopy_gpu(result: TLweSampleArray, sample: TLweSampleArray, params: TLweParams):
    thr = result.a.coefsT.thread
    thr.copy(sample.a.coefsT, dest=result.a.coefsT)
    thr.copy(sample.current_variances, dest=result.current_variances)


class TLweSymEncryptZero(Computation):

    def __init__(self, shape, noise: float, params: TLweParams, perf_params: PerformanceParameters):

        N = params.polynomial_degree
        k = params.mask_size

        self._transform_type = params.transform_type

        a = Type(Torus32, shape + (k + 1, N))
        cv = Type(numpy.float64, shape)
        key = Type(numpy.int32, (k, N))
        noises1 = Type(Torus32, shape + (k, N))
        noises2 = Type(Torus32, shape + (N,))

        self._noise = noise
        self._k = k
        self._N = N
        self._perf_params = perf_params

        Computation.__init__(self,
            [Parameter('result_a', Annotation(a, 'o')),
            Parameter('result_cv', Annotation(cv, 'o')),
            Parameter('key', Annotation(key, 'i')),
            Parameter('noises1', Annotation(noises1, 'i')),
            Parameter('noises2', Annotation(noises2, 'i')),
            ])

    def _build_plan(
            self, plan_factory, device_params,
            result_a, result_cv, key, noises1, noises2):

        plan = plan_factory()

        N = self._N

        perf_params = performance_parameters_for_device(self._perf_params, device_params)

        transform = get_transform(self._transform_type)

        ft_key = transform.ForwardTransform(key.shape[:-1], N, perf_params)
        key_tr = plan.temp_array_like(ft_key.parameter.output)

        ft_noises = transform.ForwardTransform(noises1.shape[:-1], N, perf_params)
        noises1_tr = plan.temp_array_like(ft_noises.parameter.output)

        ift = transform.InverseTransform(noises1.shape[:-1], N, perf_params)
        ift_res = plan.temp_array_like(ift.parameter.output)

        mul_tr = Transformation(
            [
                Parameter('output', Annotation(ift.parameter.input, 'o')),
                Parameter('key', Annotation(key_tr, 'i')),
                Parameter('noises1', Annotation(noises1_tr, 'i'))
            ],
            """
            ${output.store_same}(${tr_ctype}unpack(${mul}(
                ${tr_ctype}pack(${key.load_idx}(${idxs[-2]}, ${idxs[-1]})),
                ${tr_ctype}pack(${noises1.load_same})
                )));
            """,
            connectors=['output', 'noises1'],
            render_kwds=dict(
                mul=transform.transformed_mul(perf_params),
                tr_ctype=transform.transformed_internal_ctype()))

        ift.parameter.input.connect(mul_tr, mul_tr.output, key=mul_tr.key, noises1=mul_tr.noises1)

        batch_len = len(result_a.shape) - 2

        plan.computation_call(ft_key, key_tr, key)
        plan.computation_call(ft_noises, noises1_tr, noises1)
        plan.computation_call(ift, ift_res, key_tr, noises1_tr)
        plan.kernel_call(
            TEMPLATE.get_def("encrypt_zero_fill_result"),
            [result_a, result_cv, noises1, noises2, ift_res],
            global_size=(helpers.product(result_a.shape[:-2]), self._k + 1, N),
            render_kwds=dict(
                noise=self._noise, k=self._k,
                noises1_slices=(batch_len, 1, 1),
                noises2_slices=(batch_len, 1),
                cv_slices=(batch_len,)
                ))

        return plan


# create an homogeneous tlwe sample
def tLweSymEncryptZero_gpu(
        thr, rng, result: 'TLweSampleArray', noise: float, key: 'TLweKey',
        perf_params: PerformanceParameters):

    N = key.params.polynomial_degree
    k = key.params.mask_size

    noises2 = rand_gaussian_torus32(thr, rng, 0, noise, result.shape + (N,))
    noises1 = rand_uniform_torus32(thr, rng, result.shape + (k, N))

    comp = get_computation(
        thr, TLweSymEncryptZero,
        result.shape, noise, key.params, perf_params)
    comp(result.a.coefsT, result.current_variances, key.key.coefs, noises1, noises2)


# Computes the inverse FFT of the coefficients of the TLWE sample
def tLweToFFTConvert_gpu(
        thr, result: 'TLweSampleFFTArray', source: 'TLweSampleArray', params: 'TLweParams',
        perf_params: PerformanceParameters):

    perf_params = performance_parameters_for_device(perf_params, thr.device_params)

    transform = get_transform(params.transform_type)
    comp = get_computation(
        thr, transform.ForwardTransform, source.a.coefsT.shape[:-1], source.a.coefsT.shape[-1],
        perf_params)
    comp(result.a.coefsC, source.a.coefsT)
    thr.copy_array(source.current_variances, dest=result.current_variances)
