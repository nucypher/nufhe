import numpy

from .numeric_functions import Float
from .random_numbers import rand_uniform_int32
from .lwe import LweParams
from .gpu_polynomials import TorusPolynomialArray, IntPolynomialArray, LagrangeHalfCPolynomialArray


class TLweParams:

    def __init__(
            self, polynomial_degree: int, mask_size: int, min_noise: float, max_noise: float, transform_type):

        self.polynomial_degree = polynomial_degree # must be a power of 2
        self.mask_size = mask_size # number of polynomials in the mask
        self.min_noise = min_noise # minimum noise s.t. the sample is secure
        self.max_noise = max_noise # maximum noise s.t. we can decrypt
        self.extracted_lweparams = LweParams(
            polynomial_degree * mask_size, min_noise, max_noise) # lwe params if one extracts
        self.transform_type = transform_type


class TLweKey:

    def __init__(self, thr, rng, params: TLweParams):
        N = params.polynomial_degree
        k = params.mask_size

        key = IntPolynomialArray.from_array(rand_uniform_int32(thr, rng, (k, N)))

        self.params = params # the parameters of the key
        self.key = key # the key (i.e k binary polynomials)


class TLweSampleArray:

    def __init__(self, thr, params: TLweParams, shape):
        self.mask_size = params.mask_size

        # array of length k+1: mask + right term
        self.a = TorusPolynomialArray(thr, params.polynomial_degree, shape + (self.mask_size + 1,))

        # avg variance of the sample
        self.current_variances = thr.to_device(numpy.zeros(shape, Float))

        self.shape = shape


class TLweSampleFFTArray:

    def __init__(self, thr, params: TLweParams, shape):
        self.mask_size = params.mask_size

        # array of length k+1: mask + right term
        self.a = LagrangeHalfCPolynomialArray(
            thr, params.transform_type, params.polynomial_degree, shape + (self.mask_size + 1,))

        # avg variance of the sample
        self.current_variances = thr.to_device(numpy.zeros(shape, Float))

        self.shape = shape
