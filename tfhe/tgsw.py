import numpy

from .tlwe import *
from .tlwe_gpu import tLweToFFTConvert_gpu


class TGswParams:

    def __init__(self, l: int, Bgbit: int, tlwe_params: TLweParams):

        Bg = 1 << Bgbit
        halfBg = Bg // 2

        h = Torus32(1) << (32 - numpy.arange(1, l+1) * Bgbit) # 1/(Bg^(i+1)) as a Torus32

        # offset = Bg/2 * (2^(32-Bgbit) + 2^(32-2*Bgbit) + ... + 2^(32-l*Bgbit))
        offset = int64_to_int32(sum(1 << (32 - numpy.arange(1, l+1) * Bgbit)) * halfBg)

        self.l = l # decomp length
        self.Bgbit = Bgbit # log_2(Bg)
        self.Bg = Bg # decomposition base (must be a power of 2)
        self.halfBg = halfBg # Bg/2
        self.maskMod = Bg - 1 # Bg-1
        self.tlwe_params = tlwe_params # Params of each row
        self.kpl = (tlwe_params.k + 1) * l # number of rows = (k+1)*l
        self.h = h # powers of Bgbit
        self.offset = offset # offset = Bg/2 * (2^(32-Bgbit) + 2^(32-2*Bgbit) + ... + 2^(32-l*Bgbit))


class TGswKey:

    def __init__(self, thr, rng, params: TGswParams):
        self.params = params # the parameters
        self.tlwe_params = params.tlwe_params # the tlwe params of each rows
        self.tlwe_key = TLweKey(thr, rng, params.tlwe_params)


class TGswSampleArray:

    def __init__(self, thr, params: TGswParams, shape):
        self.k = params.tlwe_params.k
        self.l = params.l
        self.samples = TLweSampleArray(thr, params.tlwe_params, shape + (self.k + 1, self.l))


class TGswSampleFFTArray:

    def __init__(self, thr, params: TGswParams, shape):
        self.k = params.tlwe_params.k
        self.l = params.l
        self.samples = TLweSampleFFTArray(thr, params.tlwe_params, shape + (self.k + 1, self.l))


# For all the kpl TLWE samples composing the TGSW sample
# It computes the inverse FFT of the coefficients of the TLWE sample
def tGswToFFTConvert(thr, result: TGswSampleFFTArray, source: TGswSampleArray, params: TGswParams):
    tLweToFFTConvert_gpu(thr, result.samples, source.samples, params.tlwe_params)
