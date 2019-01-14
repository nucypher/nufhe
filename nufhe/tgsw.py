# Copyright (C) 2018 NuCypher
#
# This file is part of nufhe.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Torus Gentry, Sahai & Waters bootstrapping functions.
"""

import pickle

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
from .performance import PerformanceParametersForDevice


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

    def __eq__(self, other: 'TGswParams'):
        return (
            self.__class__ == other.__class__
            and self.decomp_length == other.decomp_length
            and self.bs_log2_base == other.bs_log2_base
            and self.tlwe_params == other.tlwe_params)

    def __hash__(self):
        return hash((self.__class__, self.decomp_length, self.bs_log2_base, self.tlwe_params))


class TGswKey:

    def __init__(self, params: TGswParams, tlwe_key: TLweKey):
        self.params = params
        self.tlwe_key = tlwe_key

    @classmethod
    def from_rng(cls, thr, params: TGswParams, rng):
        return cls(params, TLweKey.from_rng(thr, params.tlwe_params, rng))


class TGswSampleArray:

    def __init__(self, params: TGswParams, samples: TLweSampleArray):
        self.mask_size = params.tlwe_params.mask_size
        self.decomp_length = params.decomp_length
        self.samples = samples
        self.params = params
        self.shape = samples.shape[:-2]

    @classmethod
    def empty(cls, thr, params: TGswParams, shape):
        mask_size = params.tlwe_params.mask_size
        decomp_length = params.decomp_length
        samples = TLweSampleArray.empty(
            thr, params.tlwe_params, shape + (mask_size + 1, decomp_length))
        return cls(params, samples)


class TransformedTGswSampleArray:

    def __init__(self, params: TGswParams, samples: TransformedTLweSampleArray):
        self.mask_size = params.tlwe_params.mask_size
        self.decomp_length = params.decomp_length
        self.samples = samples
        self.params = params
        self.shape = samples.shape[:-2]

    @classmethod
    def empty(cls, thr, params: TGswParams, shape):
        mask_size = params.tlwe_params.mask_size
        decomp_length = params.decomp_length
        samples = TransformedTLweSampleArray.empty(
            thr, params.tlwe_params, shape + (mask_size + 1, decomp_length))
        return cls(params, samples)

    def dump(self, file_obj):
        pickle.dump(self.params, file_obj)
        self.samples.dump(file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        params = pickle.load(file_obj)
        samples = TransformedTLweSampleArray.load(file_obj, thr)
        return cls(params, samples)

    def __eq__(self, other: 'TransformedTGswSampleArray'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and self.samples == other.samples)


# For all the kpl TLWE samples composing the TGSW sample
# It computes the inverse FFT of the coefficients of the TLWE sample
def tgsw_transform_samples(
        thr, result: TransformedTGswSampleArray, source: TGswSampleArray,
        perf_params: PerformanceParametersForDevice):
    tlwe_transform_samples(thr, result.samples, source.samples, perf_params)


# result += message * H
def tgsw_add_message(thr, result: TGswSampleArray, messages):
    comp = get_computation(thr, TGswAddMessage, result.params, result.shape)
    comp(result.samples.a.coeffs, messages)


# Result = tGsw(0)
def tgsw_encrypt_zero(
        thr, rng, result: TGswSampleArray, noise: float, key: TGswKey,
        perf_params: PerformanceParametersForDevice):
    tlwe_encrypt_zero(thr, rng, result.samples, noise, key.tlwe_key, perf_params)


# encrypts a constant message
def tgsw_encrypt_int(
        thr, rng, result: TGswSampleArray, messages, noise: float, key: TGswKey,
        perf_params: PerformanceParametersForDevice):

    # TYPING: messages::Array{Int32, 1}
    tgsw_encrypt_zero(thr, rng, result, noise, key, perf_params)
    tgsw_add_message(thr, result, messages)


# external product: accum = gsw (*) accum
def tgsw_transformed_external_mul(
        thr, result: TLweSampleArray, bootstrap_key: TransformedTGswSampleArray, bk_row_idx: int,
        perf_params: PerformanceParametersForDevice):
    assert len(bootstrap_key.shape) == 1
    comp = get_computation(
        thr, TGswTransformedExternalMul,
        bootstrap_key.params, result.shape, bootstrap_key.shape[0], perf_params)
    comp(result.a.coeffs, bootstrap_key.samples.a.coeffs, bk_row_idx)
