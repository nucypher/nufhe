"""
Torus LWE functions.
"""

import numpy

from .polynomial_transform import get_transform
from .computation_cache import get_computation
from .numeric_functions import Float
from .random_numbers import rand_uniform_int32
from .lwe import LweParams, LweSampleArray
from .polynomials import (
    TorusPolynomialArray,
    IntPolynomialArray,
    TransformedPolynomialArray,
    shift_tp_minus_one_power_from_array,
    )
from .random_numbers import rand_gaussian_torus32, rand_uniform_torus32
from .performance import PerformanceParameters, performance_parameters_for_device
from .tlwe_gpu import (
    TLweNoiselessTrivial,
    TLweExtractLweSamples,
    TLweEncryptZero,
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


class TLweKey:

    def __init__(self, thr, params: TLweParams, rng):
        polynomial_degree = params.polynomial_degree
        mask_size = params.mask_size

        self.params = params

        # `mask_size` binary polynomials
        self.key = IntPolynomialArray(rand_uniform_int32(thr, rng, (mask_size, polynomial_degree)))


class TLweSampleArray:

    def __init__(self, thr, params: TLweParams, shape):

        # array of length mask size + 1: mask + right term
        self.a = TorusPolynomialArray.empty(
            thr, params.polynomial_degree, shape + (params.mask_size + 1,))

        # avg variance of the sample
        self.current_variances = thr.to_device(numpy.zeros(shape, Float))

        self.shape = shape
        self.params = params


class TransformedTLweSampleArray:

    def __init__(self, thr, params: TLweParams, shape):

        # array of length mask size + 1: mask + right term
        self.a = TransformedPolynomialArray.empty(
            thr, params.transform_type, params.polynomial_degree, shape + (params.mask_size + 1,))

        # avg variance of the sample
        self.current_variances = thr.to_device(numpy.zeros(shape, Float))

        self.shape = shape
        self.params = params


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
        perf_params: PerformanceParameters):

    polynomial_degree = key.params.polynomial_degree
    mask_size = key.params.mask_size

    noises1 = rand_uniform_torus32(thr, rng, result.shape + (mask_size, polynomial_degree))
    noises2 = rand_gaussian_torus32(thr, rng, 0, noise, result.shape + (polynomial_degree,))

    comp = get_computation(thr, TLweEncryptZero, key.params, result.shape, noise, perf_params)
    comp(result.a.coeffs, result.current_variances, key.key.coeffs, noises1, noises2)


# Computes the inverse FFT of the coefficients of the TLWE sample
def tlwe_transform_samples(
        thr, result: TransformedTLweSampleArray, source: TLweSampleArray,
        perf_params: PerformanceParameters):

    perf_params = performance_parameters_for_device(perf_params, thr.device_params)

    transform = get_transform(source.params.transform_type)
    comp = get_computation(
        thr, transform.ForwardTransform, source.a.coeffs.shape[:-1], source.a.coeffs.shape[-1],
        perf_params)
    comp(result.a.coeffs, source.a.coeffs)
    thr.copy_array(source.current_variances, dest=result.current_variances)
