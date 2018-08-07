"""
Torus Gentry, Sahai & Waters bootstrapping functions.
"""

import numpy

from .numeric_functions import Torus32
from .tlwe import (
    TLweParams,
    TLweKey,
    TLweSampleArray,
    TransformedTLweSampleArray,
    tlwe_transform_samples,
    tlwe_encrypt_zero,
    )
from .tgsw_gpu import (
    TGswAddMessage,
    TGswTransformedExternalMul,
    )
from .computation_cache import get_computation
from .performance import PerformanceParameters


class TGswParams:

    def __init__(self, tlwe_params: TLweParams, decomp_length: int, bs_log2_base: int):

        # 1/(base^(i+1)) as a Torus32
        decomp_range = numpy.arange(1, decomp_length + 1)
        self.base_powers = (2**(32 - decomp_range * bs_log2_base)).astype(Torus32)

        # offset = base/2 * Sum{j=1..decomp_length} 2^(32 - j * bs_log2_base)
        self.offset = (
            self.base_powers.astype(numpy.int64).sum() * (2**bs_log2_base // 2)).astype(Torus32)

        self.decomp_length = decomp_length
        self.bs_log2_base = bs_log2_base
        self.tlwe_params = tlwe_params # Params of each row


class TGswKey:

    def __init__(self, thr, params: TGswParams, rng):
        self.params = params
        self.tlwe_key = TLweKey(thr, params.tlwe_params, rng)


class TGswSampleArray:

    def __init__(self, thr, params: TGswParams, shape):
        self.mask_size = params.tlwe_params.mask_size
        self.decomp_length = params.decomp_length
        self.samples = TLweSampleArray(
            thr, params.tlwe_params, shape + (self.mask_size + 1, self.decomp_length))
        self.params = params
        self.shape = shape


class TransformedTGswSampleArray:

    def __init__(self, thr, params: TGswParams, shape):
        self.mask_size = params.tlwe_params.mask_size
        self.decomp_length = params.decomp_length
        self.samples = TransformedTLweSampleArray(
            thr, params.tlwe_params, shape + (self.mask_size + 1, self.decomp_length))
        self.params = params
        self.shape = shape


# For all the kpl TLWE samples composing the TGSW sample
# It computes the inverse FFT of the coefficients of the TLWE sample
def tgsw_transform_samples(
        thr, result: TransformedTGswSampleArray, source: TGswSampleArray,
        perf_params: PerformanceParameters):
    tlwe_transform_samples(thr, result.samples, source.samples, perf_params)


# result += message * H
def tgsw_add_message(thr, result: TGswSampleArray, messages):
    comp = get_computation(thr, TGswAddMessage, result.params, result.shape)
    comp(result.samples.a.coeffs, messages)


# Result = tGsw(0)
def tgsw_encrypt_zero(
        thr, rng, result: TGswSampleArray, noise: float, key: TGswKey,
        perf_params: PerformanceParameters):
    tlwe_encrypt_zero(thr, rng, result.samples, noise, key.tlwe_key, perf_params)


# encrypts a constant message
def tgsw_encrypt_int(
        thr, rng, result: TGswSampleArray, messages, noise: float, key: TGswKey,
        perf_params: PerformanceParameters):

    # TYPING: messages::Array{Int32, 1}
    tgsw_encrypt_zero(thr, rng, result, noise, key, perf_params)
    tgsw_add_message(thr, result, messages)


# external product: accum = gsw (*) accum
def tgsw_transformed_external_mul(
        thr, result: TLweSampleArray, bootstrap_key: TransformedTGswSampleArray, bk_row_idx: int,
        perf_params: PerformanceParameters):
    assert len(bootstrap_key.shape) == 1
    comp = get_computation(
        thr, TGswTransformedExternalMul,
        bootstrap_key.params, result.shape, bootstrap_key.shape[0], perf_params)
    comp(result.a.coeffs, bootstrap_key.samples.a.coeffs, bk_row_idx)
