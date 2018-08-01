"""
LWE (Learning With Errors) functions.
"""

import numpy

from reikna.core import Type

from .numeric_functions import (
    Torus32,
    Float,
    )
from .lwe_gpu import (
    LweKeySwitchTranslate_fromArray,
    LweKeySwitchKeyComputation,
    LweSymEncrypt,
    LwePhase,
    LweLinear,
    LweNoiselessTrivial,
    )
from .random_numbers import (
    rand_uniform_int32,
    rand_gaussian_float,
    rand_uniform_torus32,
    rand_gaussian_torus32,
    )
from .computation_cache import get_computation


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

    # extractions Ring Lwe * Lwe
    @classmethod
    def from_key(cls, params: LweParams, tlwe_key: 'TLweKey'):
        N = tlwe_key.params.polynomial_degree
        k = tlwe_key.params.mask_size
        assert params.size == k * N

        key = tlwe_key.key.coefs.ravel()

        return cls(params, key)


class LweSampleArrayShapeInfo:

    def __init__(self, a, b, current_variances):
        self.a = Type.from_value(a)
        self.b = Type.from_value(b)
        self.current_variances = Type.from_value(current_variances)
        self.shape = self.b.shape

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.a == other.a
            and self.b == other.b
            and self.current_variances == other.current_variances
            )

    def __hash__(self):
        return hash((self.__class__, self.a, self.b, self.current_variances))


class LweSampleArray:

    def __init__(self, params: LweParams, a, b, current_variances):
        self.params = params
        self.a = a
        self.b = b
        self.current_variances = current_variances
        self.shape_info = LweSampleArrayShapeInfo(a, b, current_variances)

    @classmethod
    def empty(cls, thr, params: LweParams, shape):
        a = thr.array(shape + (params.size,), Torus32)
        b = thr.array(shape, Torus32)
        current_variances = thr.array(shape, Float)
        return cls(params, a, b, current_variances)

    def __getitem__(self, index):
        a_view = self.a[index]
        b_view = self.b[index]
        cv_view = self.current_variances[index]
        return LweSampleArray(self.params, a_view, b_view, cv_view)


class LweKeySwitchKey:

    def __init__(self, thr, rng, n: int, t: int, basebit: int, in_key: LweKey, out_key: LweKey):
        extracted_n = n
        base = 1 << basebit
        out_params = out_key.params
        self.ks = LweSampleArray.empty(thr, out_params, (extracted_n, t, base))
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

    lweKeySwitchTranslate_fromArray_gpu(result, ks.ks, params, sample.a, sample.b, n, t, basebit)


def lweKeySwitchTranslate_fromArray_gpu(result, ks, params, a, b, outer_n, t, basebit):

    inner_n = result.a.shape[-1]
    thr = result.a.thread

    comp = get_computation(
        thr, LweKeySwitchTranslate_fromArray,
        result.shape_info, t, outer_n, inner_n, basebit)
    comp(
        result.a, result.b, result.current_variances,
        ks.a, ks.b, ks.current_variances,
        a, b)


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


def lweSymEncrypt_gpu(thr, rng, result: 'LweSampleArray', messages, alpha: float, key: 'LweKey'):
    n = key.params.size
    noises_b = rand_gaussian_torus32(thr, rng, 0, alpha, messages.shape)
    noises_a = rand_uniform_torus32(thr, rng, messages.shape + (n,))
    comp = get_computation(thr, LweSymEncrypt, messages.shape, n, alpha)
    comp(result.a, result.b, result.current_variances, messages, key.key, noises_a, noises_b)


def lwePhase_gpu(thr, sample: 'LweSampleArray', key: 'LweKey'):
    comp = get_computation(thr, LwePhase, sample.shape_info.shape, key.params.size)
    result = thr.empty_like(sample.b)
    comp(result, sample.a, sample.b, key.key)
    return result.get()


# result = (0,mu)
def lweNoiselessTrivial_gpu(thr, result: 'LweSampleArray', mu, params):
    comp = get_computation(thr, LweNoiselessTrivial, result.shape_info, params)
    comp(result.a, result.b, result.current_variances, mu)


# Arithmetic operations on LWE samples


# result = -sample
def lweNegate_gpu(thr, result: 'LweSampleArray', source: 'LweSampleArray', params):
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info, params)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, -1)


# result = sample
def lweCopy_gpu(thr, result: 'LweSampleArray', source: 'LweSampleArray', params):
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info, params)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, 1)


# result = result + sample
def lweAddTo_gpu(thr, result: 'LweSampleArray', source: 'LweSampleArray', params):
    comp = get_computation(
        thr, LweLinear, result.shape_info, source.shape_info, params, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, 1)


# result = result + p * sample
def lweAddMulTo_gpu(thr, result: 'LweSampleArray', p: int, source: 'LweSampleArray', params):
    comp = get_computation(
        thr, LweLinear, result.shape_info, source.shape_info, params, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, p)


# result = result - sample
def lweSubTo_gpu(thr, result: 'LweSampleArray', source: 'LweSampleArray', params):
    comp = get_computation(
        thr, LweLinear, result.shape_info, source.shape_info, params, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, -1)


# result = result - p * sample
def lweSubMulTo_gpu(thr, result: 'LweSampleArray', p: int, source: 'LweSampleArray', params):
    comp = get_computation(
        thr, LweLinear, result.shape_info, source.shape_info, params, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, -p)
