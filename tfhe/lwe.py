import numpy

from .numeric_functions import (
    Torus32,
    Float,
    )
from .lwe_gpu import (
    lweKeySwitchTranslate_fromArray_gpu,
    LweKeySwitchKey_gpu,
    )
from .random_numbers import (
    rand_uniform_int32,
    )


class LweParams:

    def __init__(self, size: int, alpha_min: float, alpha_max: float):
        self.size = size
        self.alpha_min = alpha_min # the smallest noise that makes it secure
        self.alpha_max = alpha_max # the biggest noise that allows decryption


class LweKey:

    def __init__(self, params: LweParams, key):
        self.params = params
        self.key = key # 1D array of Int32

    @classmethod
    def from_rng(cls, thr, rng, params: LweParams):
        return cls(params, rand_uniform_int32(thr, rng, (params.size,)))

    # extractions Ring Lwe . Lwe
    @classmethod
    def from_key(cls, params: LweParams, tlwe_key):  # sans doute un param supplémentaire
        # TYPING: tlwe_key: TLweKey
        N = tlwe_key.params.polynomial_degree
        k = tlwe_key.params.mask_size
        assert params.size == k * N

        key = tlwe_key.key.coefs.ravel()

        return cls(params, key)


class LweSampleArray:

    def __init__(self, thr, params: LweParams, shape):
        self.a = thr.array(shape + (params.size,), Torus32)
        self.b = thr.array(shape, Torus32)
        self.current_variances = thr.array(shape, Float)
        self.shape = shape
        self.params = params


class LweKeySwitchKey:

    def __init__(self, thr, rng, n: int, t: int, basebit: int, in_key: LweKey, out_key: LweKey):
        extracted_n = n
        base = 1 << basebit
        out_params = out_key.params
        self.ks = LweSampleArray(thr, out_params, (extracted_n, t, base))
        LweKeySwitchKey_gpu(
            thr, rng, self.ks, extracted_n, t, basebit, in_key, out_key)

        self.input_size = n # length of the input key: s'
        self.t = t # decomposition length
        self.basebit = basebit # log_2(base)
        self.base = base # decomposition base: a power of 2
        self.out_params = out_params # params of the output key s


#sample=(a',b')
def lweKeySwitch(thr, result: LweSampleArray, ks: LweKeySwitchKey, sample: LweSampleArray):

    params = ks.out_params
    n = ks.input_size
    basebit = ks.basebit
    t = ks.t

    lweNoiselessTrivial(thr, result, sample.b, params)
    lweKeySwitchTranslate_fromArray_gpu(result, ks.ks, params, sample.a, n, t, basebit)


# Arithmetic operations on Lwe samples

# result = sample
def lweCopy(result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    result.a = sample.a.copy()
    result.b = sample.b.copy()
    result.current_variances = sample.current_variances.copy()


# result = -sample
def lweNegate(result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    result.a = -sample.a
    result.b = -sample.b
    result.current_variances = sample.current_variances.copy()


# result = result + p.sample
def lweAddMulTo(result: LweSampleArray, p: numpy.int32, sample: LweSampleArray, params: LweParams):
    result.a += p * sample.a
    result.b += p * sample.b
    result.current_variances += p**2 * sample.current_variances


# result = result - p.sample
def lweSubMulTo(result: LweSampleArray, p: numpy.int32, sample: LweSampleArray, params: LweParams):
    result.a -= p * sample.a
    result.b -= p * sample.b
    result.current_variances += p**2 * sample.current_variances


# result = result + sample
def lweAddTo(thr, result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    # GPU: array operations or a custom kernel
    result.a += sample.a
    result.b += sample.b
    result.current_variances += sample.current_variances


# result = result - sample
def lweSubTo(thr, result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    result.a -= sample.a
    result.b -= sample.b
    result.current_variances += sample.current_variances


# result = (0,mu)
def lweNoiselessTrivial(thr, result: LweSampleArray, mus, params: LweParams):
    # TYPING: mus: Union{Array{Torus32}, Torus32}
    # GPU: array operations
    result.a.fill(0)
    if isinstance(mus, numpy.ndarray):
        raise NotImplementedError()
    elif hasattr(mus, 'thread'):
        thr.copy_array(mus, dest=result.b)
    else:
        result.b.fill(mus)
    result.current_variances.fill(0)
