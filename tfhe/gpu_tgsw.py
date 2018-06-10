import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.fft import FFT
from reikna.algorithms import PureParallel
import reikna.transformations as transformations
from reikna.cluda import dtypes, functions

from .gpu_polynomials import TorusPolynomialArray
from .tgsw import TGswParams, TGswSampleArray, TGswSampleFFTArray
from .tlwe import TLweSampleArray
from .polynomial_transform import (
    ForwardTransform, InverseTransform, transformed_dtype, transformed_length,
    transformed_mul, transformed_add)
from .computation_cache import get_computation


def get_TGswTorus32PolynomialDecompH_trf(result, params: TGswParams):
    sample = Type(result.dtype, result.shape[:-2] + result.shape[-1:])
    return Transformation(
        [Parameter('output', Annotation(result, 'o')),
        Parameter('sample', Annotation(sample, 'i'))],
        """
        ${sample.ctype} sample = ${sample.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-1]});
        int p = ${idxs[-2]} + 1;
        int decal = 32 - p * ${params.Bgbit};
        ${output.store_same}(
            (((sample + (${params.offset})) >> decal) & ${params.maskMod}) - ${params.halfBg}
        );
        """,
        connectors=['output'],
        render_kwds=dict(params=params))


def get_TLweFFTAddMulRTo_trf(tmpa_a, gsw):
    N = tmpa_a.shape[-1] * 2
    k = tmpa_a.shape[-2] - 1
    l = gsw.shape[-3]
    batch = tmpa_a.shape[:-2]
    decaFFT = Type(tmpa_a.dtype, batch + (k + 1, l, N//2))

    return Transformation(
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
            tmpa_a = ${add}(tmpa_a, ${mul}(a, b));
        }
        %endfor
        %endfor

        ${tmpa_a.store_same}(tmpa_a);
        """,
        connectors=['tmpa_a'],
        render_kwds=dict(k=k, l=l, add=transformed_add(), mul=transformed_mul()))


class TGswFFTExternMulToTLwe(Computation):

    def __init__(self, accum_a, gsw, params: TGswParams):
        tlwe_params = params.tlwe_params
        k = tlwe_params.k
        l = params.l
        N = tlwe_params.N

        batch_shape = accum_a.shape[:-2]

        tdtype = transformed_dtype()
        tlength = transformed_length(N)

        deca_shape = batch_shape + (k + 1, l)
        tmpa_shape = batch_shape + (k + 1,)

        self._deca_type = Type(numpy.int32, deca_shape + (N,))
        self._deca_fft_type = Type(tdtype, deca_shape + (tlength,))
        self._tmpa_a_type = Type(tdtype, tmpa_shape + (tlength,))

        decomp = get_TGswTorus32PolynomialDecompH_trf(self._deca_type, params)
        self._ip_ifft = ForwardTransform(deca_shape, N)
        self._ip_ifft.parameter.input.connect(
            decomp, decomp.output, sample=decomp.sample)

        add = get_TLweFFTAddMulRTo_trf(self._tmpa_a_type, gsw)
        self._tp_fft = InverseTransform(tmpa_shape, N)
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
        result: TLweSampleArray, bki: TGswSampleFFTArray, bk_idx: int, bk_params: TGswParams):

    thr = result.a.coefsT.thread
    comp = get_computation(thr, TGswFFTExternMulToTLwe, result.a.coefsT, bki.samples.a.coefsC, bk_params)
    comp(result.a.coefsT, bki.samples.a.coefsC, bk_idx)
