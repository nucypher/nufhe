import numpy

from .lwe import *
from .polynomials import *


class TLweParams:

    def __init__(self, N: int, k: int, alpha_min: float, alpha_max: float):
        self.N = N # a power of 2: degree of the polynomials
        self.k = k # number of polynomials in the mask
        self.alpha_min = alpha_min # minimal noise s.t. the sample is secure
        self.alpha_max = alpha_max # maximal noise s.t. we can decrypt
        self.extracted_lweparams = LweParams(N * k, alpha_min, alpha_max) # lwe params if one extracts


class TLweKey:

    def __init__(self, rng, params: TLweParams):
        N = params.N
        k = params.k
        # GPU: array operation or RNG on device
        key = IntPolynomialArray(N, (k,))
        key.coefs[:,:] = rand_uniform_int32(rng, (k, N))

        self.params = params # the parameters of the key
        self.key = key # the key (i.e k binary polynomials)


class TLweSampleArray:

    def __init__(self, params: TLweParams, shape):
        self.k = params.k

        # array of length k+1: mask + right term
        self.a = TorusPolynomialArray(params.N, shape + (self.k + 1,))

        # avg variance of the sample
        self.current_variances = numpy.zeros(shape, numpy.float64)

        self.shape = shape


class TLweSampleFFTArray:

    def __init__(self, params: TLweParams, shape):
        self.k = params.k

        # array of length k+1: mask + right term
        self.a = LagrangeHalfCPolynomialArray(params.N, shape + (self.k + 1,))

        # avg variance of the sample
        self.current_variances = numpy.zeros(shape, numpy.float64)

        self.shape = shape


def tLweExtractLweSampleIndex(
        result: LweSampleArray, x: TLweSampleArray, index: int, params: LweParams, rparams: TLweParams):

    N = rparams.N
    k = rparams.k
    assert params.n == k*N

    # TODO: use an appropriate method to get coefsT
    a_view = result.a.reshape(result.shape + (k, N))
    a_view[:,:,:(index+1)] = x.a.coefsT[:, :k, index::-1]
    a_view[:,:,(index+1):] = -x.a.coefsT[:, :k, :index:-1]

    numpy.copyto(result.b, x.a.coefsT[:, k, index])


def tLweExtractLweSample(result: LweSampleArray, x: TLweSampleArray, params: LweParams, rparams: TLweParams):
    tLweExtractLweSampleIndex(result, x, 0, params, rparams)


# create an homogeneous tlwe sample
def tLweSymEncryptZero(rng, result: TLweSampleArray, alpha: float, key: TLweKey):
    N = key.params.N
    k = key.params.k

    # TODO: use an appropriate method

    result.a.coefsT[:,:,:,k,:] = rand_gaussian_torus32(rng, 0, alpha, result.shape + (N,))

    result.a.coefsT[:,:,:,:k,:] = rand_uniform_torus32(rng, result.shape + (k, N))

    tmp1 = LagrangeHalfCPolynomialArray(N, key.key.shape)
    tmp2 = LagrangeHalfCPolynomialArray(N, result.shape + (k,))
    tmp3 = LagrangeHalfCPolynomialArray(N, result.shape + (k,))
    tmpr = TorusPolynomialArray(N, result.shape + (k,))

    ip_ifft_(tmp1, key.key)
    tp_ifft_(tmp2, TorusPolynomialArray.from_arr(result.a.coefsT[:, :, :, :k, :]))
    lp_mul_(tmp3, tmp1, tmp2)
    tp_fft_(tmpr, tmp3)

    for i in range(k):
        result.a.coefsT[:,:,:,k,:] += tmpr.coefsT[:,:,:,i,:]

    result.current_variances.fill(alpha**2)


# Arithmetic operations on TLwe samples

# result = sample
def tLweCopy(result: TLweSampleArray, sample: TLweSampleArray, params: TLweParams):
    # GPU: array operations or a custom kernel
    numpy.copyto(result.a.coefsT, sample.a.coefsT) # TODO: use an appropriate method?
    numpy.copyto(result.current_variances, sample.current_variances)


# result = (0,mu)
def tLweNoiselessTrivial(result: TLweSampleArray, mu: TorusPolynomialArray, params: TLweParams):
    # GPU: array operations or a custom kernel
    k = params.k
    tp_clear_(result.a)
    result.a.coefsT[:,result.k,:] = mu.coefsT # TODO: wrap in a function?
    result.current_variances.fill(0.)


# result = result + sample
def tLweAddTo(result: TLweSampleArray, sample: TLweSampleArray, params: TLweParams):
    # GPU: array operations or a custom kernel
    k = params.k
    tp_add_to_(result.a, sample.a)
    result.current_variances += sample.current_variances


# mult externe de X^ai-1 par bki
def tLweMulByXaiMinusOne(result:TLweSampleArray, ai, bk: TLweSampleArray, params: TLweParams):
    # TYPING: ai::Array{Int32}
    tp_mul_by_xai_minus_one_(result.a, ai, bk.a)


# Computes the inverse FFT of the coefficients of the TLWE sample
def tLweToFFTConvert(result: TLweSampleFFTArray, source: TLweSampleArray, params: TLweParams):
    tp_ifft_(result.a, source.a)
    numpy.copyto(result.current_variances, source.current_variances)


# Computes the FFT of the coefficients of the TLWEfft sample
def tLweFromFFTConvert(result: TLweSampleArray, source: TLweSampleFFTArray, params: TLweParams):
    tp_fft_(result.a, source.a)
    numpy.copyto(result.current_variances, source.current_variances)

# Arithmetic operations on TLwe samples

# result = (0,0)
def tLweFFTClear(result: TLweSampleFFTArray, params: TLweParams):
    lp_clear_(result.a)
    result.current_variances.fill(0.)
