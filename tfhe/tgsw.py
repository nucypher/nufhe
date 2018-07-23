import numpy

from .numeric_functions import Torus32
from .tlwe import TLweParams, TLweKey, TLweSampleArray, TLweSampleFFTArray
from .tlwe_gpu import tLweToFFTConvert_gpu


class TGswParams:

    def __init__(self, decomp_length: int, bs_log2_base: int, tlwe_params: TLweParams):

        # 1/(base^(i+1)) as a Torus32
        decomp_range = numpy.arange(1, decomp_length+1)
        self.base_powers = (2**(32 - decomp_range * bs_log2_base)).astype(Torus32)

        # offset = base/2 * Sum{j=1..decomp_length} 2^(32 - j * bs_log2_base)
        self.offset = (
            self.base_powers.astype(numpy.int64).sum() * (2**bs_log2_base // 2)).astype(numpy.int32)

        self.decomp_length = decomp_length
        self.bs_log2_base = bs_log2_base
        self.tlwe_params = tlwe_params # Params of each row


class TGswKey:

    def __init__(self, thr, rng, params: TGswParams):
        self.params = params # the parameters
        self.tlwe_params = params.tlwe_params # the tlwe params of each rows
        self.tlwe_key = TLweKey(thr, rng, params.tlwe_params)


class TGswSampleArray:

    def __init__(self, thr, params: TGswParams, shape):
        self.mask_size = params.tlwe_params.mask_size
        self.decomp_length = params.decomp_length
        self.samples = TLweSampleArray(
            thr, params.tlwe_params, shape + (self.mask_size + 1, self.decomp_length))


class TGswSampleFFTArray:

    def __init__(self, thr, params: TGswParams, shape):
        self.mask_size = params.tlwe_params.mask_size
        self.decomp_length = params.decomp_length
        self.samples = TLweSampleFFTArray(
            thr, params.tlwe_params, shape + (self.mask_size + 1, self.decomp_length))


# For all the kpl TLWE samples composing the TGSW sample
# It computes the inverse FFT of the coefficients of the TLWE sample
def tGswToFFTConvert(thr, result: TGswSampleFFTArray, source: TGswSampleArray, params: TGswParams):
    tLweToFFTConvert_gpu(thr, result.samples, source.samples, params.tlwe_params)
