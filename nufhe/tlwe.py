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
Torus LWE functions.
"""

import pickle

import numpy

from .utils import arrays_equal
from .polynomial_transform import get_transform
from .computation_cache import get_computation
from .numeric_functions import ErrorFloat
from .random_numbers import rand_uniform_bool
from .lwe import LweParams, LweSampleArray
from .polynomials import (
    TorusPolynomialArray,
    IntPolynomialArray,
    TransformedPolynomialArray,
    shift_tp_minus_one_power_from_array,
    )
from .random_numbers import rand_gaussian_torus32, rand_uniform_torus32
from .performance import PerformanceParametersForDevice
from .tlwe_gpu import (
    TLweNoiselessTrivial,
    TLweExtractLweSamples,
    TLweEncryptZero,
    TLweTransformSamples,
    )


class TLweParams:

    def __init__(
            self, polynomial_degree: int, mask_size: int,
            min_noise: float, max_noise: float, transform_type):

        self.polynomial_degree = polynomial_degree # must be a power of 2
        self.mask_size = mask_size # number of polynomials in the mask
        self.min_noise = min_noise # minimum noise s.t. the sample is secure
        self.max_noise = max_noise # maximum noise s.t. we can decrypt
        self.extracted_lweparams = LweParams(
            polynomial_degree * mask_size, min_noise, max_noise) # LWE params if one extracts
        self.transform_type = transform_type

    def __eq__(self, other: 'TLweParams'):
        return (
            self.__class__ == other.__class__
            and self.polynomial_degree == other.polynomial_degree
            and self.mask_size == other.mask_size
            and self.min_noise == other.min_noise
            and self.max_noise == other.max_noise
            and self.transform_type == other.transform_type)

    def __hash__(self):
        return hash((
            self.__class__, self.polynomial_degree, self.mask_size,
            self.min_noise, self.max_noise, self.transform_type))


class TLweKey:

    def __init__(self, params: TLweParams, key):
        self.params = params
        self.key = key

    @classmethod
    def from_rng(cls, thr, params: TLweParams, rng):
        polynomial_degree = params.polynomial_degree
        mask_size = params.mask_size

        # `mask_size` binary polynomials
        key = IntPolynomialArray(rand_uniform_bool(thr, rng, (mask_size, polynomial_degree)))

        return cls(params, key)


class TLweSampleArray:

    def __init__(self, params: TLweParams, a, current_variances):
        self.a = a
        self.current_variances = current_variances
        self.shape = current_variances.shape
        self.params = params

    @classmethod
    def empty(cls, thr, params: TLweParams, shape):

        # array of length mask size + 1: mask + right term
        a = TorusPolynomialArray.empty(
            thr, params.polynomial_degree, shape + (params.mask_size + 1,))

        # avg variance of the sample
        current_variances = thr.to_device(numpy.zeros(shape, ErrorFloat))

        return cls(params, a, current_variances)


class TransformedTLweSampleArray:

    def __init__(self, params: TLweParams, a, current_variances):
        self.a = a
        self.current_variances = current_variances
        self.shape = current_variances.shape
        self.params = params

    @classmethod
    def empty(cls, thr, params: TLweParams, shape):

        # array of length mask size + 1: mask + right term
        a = TransformedPolynomialArray.empty(
            thr, params.transform_type, params.polynomial_degree, shape + (params.mask_size + 1,))

        # avg variance of the sample
        current_variances = thr.to_device(numpy.zeros(shape, ErrorFloat))

        return cls(params, a, current_variances)

    def dump(self, file_obj):
        pickle.dump(self.params, file_obj)
        self.a.dump(file_obj)
        pickle.dump(self.current_variances.get(), file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        params = pickle.load(file_obj)
        a = TransformedPolynomialArray.load(file_obj, thr)
        current_variances = pickle.load(file_obj)
        return cls(params, a, thr.to_device(current_variances))

    def __eq__(self, other: 'TransformedTLweSampleArray'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and self.a == other.a
            and arrays_equal(self.current_variances, other.current_variances))


# result = (0,mu)
def tlwe_noiseless_trivial(thr, result: TLweSampleArray, mu: TorusPolynomialArray):
    comp = get_computation(thr, TLweNoiselessTrivial, result.params, result.shape)
    comp(result.a.coeffs, result.current_variances, mu.coeffs)


def tlwe_extract_lwe_samples(thr, result: LweSampleArray, x: TLweSampleArray):
    comp = get_computation(thr, TLweExtractLweSamples, x.params, x.shape)
    # Note: `current_variances` is not filled in original TFHE
    comp(result.a, result.b, x.a.coeffs)


# mult externe de X^ai-1 par bki
def tlwe_shift_polynomials(thr, result: TLweSampleArray, bk: TLweSampleArray, powers, powers_idx):
    shift_tp_minus_one_power_from_array(thr, result.a, powers, powers_idx, bk.a)


# result = result + sample
def tlwe_add_to(thr, result: TLweSampleArray, source: TLweSampleArray):
    result.a.coeffs += source.a.coeffs
    result.current_variances += source.current_variances


# result = sample
def tlwe_copy(thr, result: TLweSampleArray, source: TLweSampleArray):
    thr.copy(source.a.coeffs, dest=result.a.coeffs)
    thr.copy(source.current_variances, dest=result.current_variances)


# create an homogeneous tlwe sample
def tlwe_encrypt_zero(
        thr, rng, result: TLweSampleArray, noise: float, key: TLweKey,
        perf_params: PerformanceParametersForDevice):

    polynomial_degree = key.params.polynomial_degree
    mask_size = key.params.mask_size

    noises1 = rand_uniform_torus32(thr, rng, result.shape + (mask_size, polynomial_degree))
    noises2 = rand_gaussian_torus32(thr, rng, 0, noise, result.shape + (polynomial_degree,))

    comp = get_computation(thr, TLweEncryptZero, key.params, result.shape, noise, perf_params)
    comp(result.a.coeffs, result.current_variances, key.key.coeffs, noises1, noises2)


# Computes the inverse FFT of the coefficients of the TLWE sample
def tlwe_transform_samples(
        thr, result: TransformedTLweSampleArray, source: TLweSampleArray,
        perf_params: PerformanceParametersForDevice):

    comp = get_computation(
        thr, TLweTransformSamples, source.params, source.a.coeffs.shape, perf_params)
    comp(result.a.coeffs, source.a.coeffs)
    thr.copy_array(source.current_variances, dest=result.current_variances)
