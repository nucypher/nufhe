import numpy

from .random_numbers import rand_uniform_int32
from .lwe import *
from .gpu_polynomials import TorusPolynomialArray, IntPolynomialArray, LagrangeHalfCPolynomialArray


class TLweParams:

    def __init__(self, N: int, k: int, alpha_min: float, alpha_max: float):
        self.N = N # a power of 2: degree of the polynomials
        self.k = k # number of polynomials in the mask
        self.alpha_min = alpha_min # minimal noise s.t. the sample is secure
        self.alpha_max = alpha_max # maximal noise s.t. we can decrypt
        self.extracted_lweparams = LweParams(N * k, alpha_min, alpha_max) # lwe params if one extracts


class TLweKey:

    def __init__(self, thr, rng, params: TLweParams):
        N = params.N
        k = params.k

        key = IntPolynomialArray.from_array(rand_uniform_int32(thr, rng, (k, N)))

        self.params = params # the parameters of the key
        self.key = key # the key (i.e k binary polynomials)


class TLweSampleArray:

    def __init__(self, thr, params: TLweParams, shape):
        self.k = params.k

        # array of length k+1: mask + right term
        self.a = TorusPolynomialArray(thr, params.N, shape + (self.k + 1,))

        # avg variance of the sample
        self.current_variances = thr.to_device(numpy.zeros(shape, Float))

        self.shape = shape


class TLweSampleFFTArray:

    def __init__(self, thr, params: TLweParams, shape):
        self.k = params.k

        # array of length k+1: mask + right term
        self.a = LagrangeHalfCPolynomialArray(thr, params.N, shape + (self.k + 1,))

        # avg variance of the sample
        self.current_variances = thr.to_device(numpy.zeros(shape, Float))

        self.shape = shape
