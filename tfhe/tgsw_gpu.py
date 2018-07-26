import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
import reikna.helpers as helpers

from .numeric_functions import Torus32
from .tlwe_gpu import tLweSymEncryptZero_gpu
from .tgsw import TGswParams, TGswSampleArray, TGswSampleFFTArray
from .tlwe import TLweSampleArray
from .computation_cache import get_computation
from .polynomial_transform import get_transform
from .performance import PerformanceParameters


TEMPLATE = helpers.template_for(__file__)


def get_TGswTorus32PolynomialDecompH_trf(result, params: TGswParams):
    sample = Type(result.dtype, result.shape[:-2] + result.shape[-1:])
    return Transformation(
        [Parameter('output', Annotation(result, 'o')),
        Parameter('sample', Annotation(sample, 'i'))],
        """
        <%
            mask = 2**params.bs_log2_base - 1
            half_base = 2**(params.bs_log2_base - 1)
        %>
        ${sample.ctype} sample = ${sample.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-1]});
        int p = ${idxs[-2]} + 1;
        int decal = 32 - p * ${params.bs_log2_base};
        ${output.store_same}(
            (((sample + (${params.offset})) >> decal) & ${mask}) - ${half_base}
        );
        """,
        connectors=['output'],
        render_kwds=dict(params=params))


def get_TLweFFTAddMulRTo_trf(
        N, transform_type, tmpa_a, gsw, tr_ctype, perf_params: PerformanceParameters):

    k = tmpa_a.shape[-2] - 1
    l = gsw.shape[-3]
    batch = tmpa_a.shape[:-2]
    transform = get_transform(transform_type)
    decaFFT = Type(tmpa_a.dtype, batch + (k + 1, l, transform.transformed_length(N)))

    return Transformation(
        [Parameter('tmpa_a', Annotation(tmpa_a, 'o')),
        Parameter('decaFFT', Annotation(decaFFT, 'i')),
        Parameter('gsw', Annotation(gsw, 'i')),
        Parameter('bk_idx', Annotation(numpy.int32))],
        """
        ${tr_ctype} tmpa_a = ${tr_ctype}pack(${dtypes.c_constant(0, tmpa_a.dtype)});

        %for i in range(k + 1):
        %for j in range(l):
        {
            ${tr_ctype} a = ${tr_ctype}pack(
                ${decaFFT.load_idx}(
                    ${", ".join(idxs[:-2])}, ${i}, ${j}, ${idxs[-1]})
                );
            ${tr_ctype} b = ${tr_ctype}pack(
                ${gsw.load_idx}(
                    ${bk_idx}, ${i}, ${j}, ${idxs[-2]}, ${idxs[-1]})
                );
            tmpa_a = ${add}(tmpa_a, ${mul}(a, b));
        }
        %endfor
        %endfor

        ${tmpa_a.store_same}(${tr_ctype}unpack(tmpa_a));
        """,
        connectors=['tmpa_a'],
        render_kwds=dict(
            k=k, l=l,
            add=transform.transformed_add(perf_params),
            mul=transform.transformed_mul(perf_params),
            tr_ctype=tr_ctype))


class TGswFFTExternMulToTLwe(Computation):

    def __init__(self, accum_a, gsw, params: TGswParams, perf_params: PerformanceParameters):
        tlwe_params = params.tlwe_params
        k = tlwe_params.mask_size
        l = params.decomp_length
        N = tlwe_params.polynomial_degree

        batch_shape = accum_a.shape[:-2]

        transform = get_transform(tlwe_params.transform_type)

        tdtype = transform.transformed_dtype()
        tlength = transform.transformed_length(N)

        deca_shape = batch_shape + (k + 1, l)
        tmpa_shape = batch_shape + (k + 1,)

        self._deca_type = Type(numpy.int32, deca_shape + (N,))
        self._deca_fft_type = Type(tdtype, deca_shape + (tlength,))
        self._tmpa_a_type = Type(tdtype, tmpa_shape + (tlength,))

        decomp = get_TGswTorus32PolynomialDecompH_trf(self._deca_type, params)
        self._ip_ifft = transform.ForwardTransform(deca_shape, N, perf_params)
        self._ip_ifft.parameter.input.connect(
            decomp, decomp.output, sample=decomp.sample)

        add = get_TLweFFTAddMulRTo_trf(
            N, tlwe_params.transform_type, self._tmpa_a_type, gsw,
            transform.transformed_internal_ctype(), perf_params)
        self._tp_fft = transform.InverseTransform(tmpa_shape, N, perf_params)
        self._tp_fft.parameter.input.connect(
            add, add.tmpa_a, decaFFT=add.decaFFT, gsw=add.gsw, bk_idx=add.bk_idx)

        Computation.__init__(self,
            [Parameter('accum_a', Annotation(accum_a, 'io')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('bk_idx', Annotation(numpy.int32))])


    def _build_plan(self, plan_factory, device_params, accum_a, gsw, bk_idx):
        plan = plan_factory()

        decaFFT = plan.temp_array_like(self._deca_fft_type)

        plan.computation_call(self._ip_ifft, decaFFT, accum_a)
        plan.computation_call(self._tp_fft, accum_a, decaFFT, gsw, bk_idx)

        return plan


def tGswFFTExternMulToTLwe_gpu(
        result: TLweSampleArray, bki: TGswSampleFFTArray, bk_idx: int, bk_params: TGswParams,
        perf_params: PerformanceParameters):

    thr = result.a.coefsT.thread
    comp = get_computation(
        thr, TGswFFTExternMulToTLwe, result.a.coefsT, bki.samples.a.coefsC, bk_params, perf_params)
    comp(result.a.coefsT, bki.samples.a.coefsC, bk_idx)


class TGswAddMuIntH(Computation):

    def __init__(self, n, params: 'TGswParams'):

        self._params = params

        k = params.tlwe_params.mask_size
        l = params.decomp_length
        N = params.tlwe_params.polynomial_degree

        result_a = Type(Torus32, (n, k + 1, l, k + 1, N))
        messages = Type(Torus32, (n,))

        self._n = n

        Computation.__init__(self,
            [Parameter('result_a', Annotation(result_a, 'o')),
            Parameter('messages', Annotation(messages, 'i'))])

    def _build_plan(self, plan_factory, device_params, result_a, messages):
        plan = plan_factory()

        plan.kernel_call(
            TEMPLATE.get_def("TGswAddMuIntH"),
            [result_a, messages],
            global_size=(self._n,),
            render_kwds=dict(
                h=self._params.base_powers,
                l=self._params.decomp_length,
                k=self._params.tlwe_params.mask_size))

        return plan


def tGswAddMuIntH_gpu(thr, result: 'TGswSampleArray', messages, params: 'TGswParams'):
    n = result.samples.a.coefsT.shape[0] # TODO: get from the parameters
    comp = get_computation(thr, TGswAddMuIntH, n, params)
    comp(result.samples.a.coefsT, messages)


# Result = tGsw(0)
def tGswEncryptZero(
        thr, rng, result: TGswSampleArray, alpha: float, key: 'TGswKey',
        perf_params: PerformanceParameters):
    rlkey = key.tlwe_key
    tLweSymEncryptZero_gpu(thr, rng, result.samples, alpha, rlkey, perf_params)


# encrypts a constant message
def tGswSymEncryptInt_gpu(
        thr, rng, result: TGswSampleArray, messages, alpha: float, key: 'TGswKey',
        perf_params: PerformanceParameters):

    # TYPING: messages::Array{Int32, 1}
    tGswEncryptZero(thr, rng, result, alpha, key, perf_params)
    tGswAddMuIntH_gpu(thr, result, messages, key.params)
