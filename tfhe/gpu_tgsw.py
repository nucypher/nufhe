import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.fft import FFT
from reikna.algorithms import PureParallel
import reikna.transformations as transformations
from reikna.cluda import dtypes, functions

from .polynomials import TorusPolynomialArray
from .tgsw import TGswParams, TGswSampleArray, TGswSampleFFTArray
from .tlwe import TLweSampleArray
from .numeric_functions import Complex
from .gpu_polynomials import I2C_FFT, I2C_FFT_v2, I2C_FFT_v3, C2I_FFT, C2I_FFT_v2
from .computation_cache import get_computation


class TGswTorus32PolynomialDecompH(PureParallel):

    def __init__(self, result, params: TGswParams):

        sample = Type(result.dtype, result.shape[:-2] + result.shape[-1:])

        PureParallel.__init__(
            self,
            [Parameter('result', Annotation(result, 'o')),
            Parameter('sample', Annotation(sample, 'i'))],
            """
            ${sample.ctype} sample = ${sample.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-1]});
            int p = ${idxs[-2]} + 1;
            int decal = 32 - p * ${params.Bgbit};
            ${result.store_same}(
                (((sample + (${params.offset})) >> decal) & ${params.maskMod}) - ${params.halfBg}
            );
            """,
            render_kwds=dict(params=params))


class TLweFFTAddMulRTo(PureParallel):

    def __init__(self, tmpa_a, gsw):
        # tmpa_a: Complex, (batch, k+1, N//2)
        # decaFFT: Complex, (batch, k+1, l, N//2)
        # gsw: Complex, (n, k+1, l, k+1, N//2)

        N = tmpa_a.shape[-1] * 2
        k = tmpa_a.shape[-2] - 1
        l = gsw.shape[-3]
        batch = tmpa_a.shape[:-2]
        decaFFT = Type(tmpa_a.dtype, batch + (k + 1, l, N//2))

        PureParallel.__init__(
            self,
            [Parameter('tmpa_a', Annotation(tmpa_a, 'o')),
            Parameter('decaFFT', Annotation(decaFFT, 'i')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('bk_idx', Annotation(numpy.int32))],
            """
            ${tmpa_a.ctype} tmpa_a = ${dtypes.c_constant(0, tmpa_a.dtype)};
            %for i in range(k + 1):
            %for j in range(l):
            {
                ${decaFFT.ctype} a = ${decaFFT.load_idx}(
                    ${", ".join(idxs[:-2])}, ${i}, ${j}, ${idxs[-1]});
                ${gsw.ctype} b = ${gsw.load_idx}(
                    ${bk_idx}, ${i}, ${j}, ${idxs[-2]}, ${idxs[-1]});
                tmpa_a = tmpa_a + ${mul}(a, b);
            }
            %endfor
            %endfor

            ${tmpa_a.store_same}(tmpa_a);
            """,
            render_kwds=dict(k=k, l=l, mul=functions.mul(tmpa_a.dtype, tmpa_a.dtype)))


class TGswFFTExternMulToTLwe(Computation):

    def __init__(self, accum_a, gsw, params: TGswParams):
        tlwe_params = params.tlwe_params
        k = tlwe_params.k
        l = params.l
        N = tlwe_params.N

        batch_shape = accum_a.shape[:-2]

        self._deca_type = Type(numpy.int32, batch_shape + (k + 1, l, N))
        self._deca_fft_type = Type(Complex, batch_shape + (k + 1, l, N // 2))
        self._tmpa_a_type = Type(Complex, batch_shape + (k + 1, N // 2))

        # TODO: can be made a transformation for the ip_ifft
        self._tGswTorus32PolynomialDecompH = TGswTorus32PolynomialDecompH(self._deca_type, params)
        self._ip_ifft = I2C_FFT_v2(self._deca_type, 2)
        self._tLweFFTAddMulRTo = TLweFFTAddMulRTo(self._tmpa_a_type, gsw)
        self._tp_fft = C2I_FFT_v2(self._tmpa_a_type)

        Computation.__init__(self,
            [Parameter('accum_a', Annotation(accum_a, 'io')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('bk_idx', Annotation(numpy.int32))])


    def _build_plan(self, plan_factory, device_params, accum_a, gsw, bk_idx):
        plan = plan_factory()

        tmpa_a = plan.temp_array_like(self._tmpa_a_type)
        deca = plan.temp_array_like(self._deca_type)
        decaFFT = plan.temp_array_like(self._deca_fft_type)

        plan.computation_call(self._tGswTorus32PolynomialDecompH, deca, accum_a)
        plan.computation_call(self._ip_ifft, decaFFT, deca)
        plan.computation_call(self._tLweFFTAddMulRTo, tmpa_a, decaFFT, gsw, bk_idx)
        plan.computation_call(self._tp_fft, accum_a, tmpa_a)

        return plan


def tGswFFTExternMulToTLwe_gpu(
        result: TLweSampleArray, bki: TGswSampleFFTArray, bk_idx: int, bk_params: TGswParams):

    thr = result.a.coefsT.thread
    comp = get_computation(thr, TGswFFTExternMulToTLwe, result.a.coefsT, bki.samples.a.coefsC, bk_params)
    comp(result.a.coefsT, bki.samples.a.coefsC, bk_idx)
