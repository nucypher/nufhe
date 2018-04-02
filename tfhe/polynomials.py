import numpy

from .numeric_functions import *


# This structure represents an integer polynomial modulo X^N+1
class IntPolynomialArray:
    def __init__(self, N, shape):
        self.coefs = numpy.empty(shape + (N,), numpy.int32)
        self.polynomial_size = N
        self.shape = shape


# This structure represents an torus polynomial modulo X^N+1
class TorusPolynomialArray:
    def __init__(self, N, shape):
        self.coefsT = numpy.empty(shape + (N,), Torus32)
        self.polynomial_size = N
        self.shape = shape

    @classmethod
    def from_arr(cls, arr):
        obj = cls(arr.shape[-1], arr.shape[:-1])
        obj.coefsT = arr
        return obj


# This structure is used for FFT operations, and is a representation
# over C of a polynomial in R[X]/X^N+1
class LagrangeHalfCPolynomialArray:
    def __init__(self, N, shape):
        assert N % 2 == 0
        self.coefsC = numpy.empty(shape + (N // 2,), numpy.complex128)
        self.polynomial_size = N
        self.shape = shape


def _coefs(p):
    # TODO: different field names help with debugging, remove later
    if type(p) == IntPolynomialArray:
        return p.coefs
    elif type(p) == TorusPolynomialArray:
        return p.coefsT
    else:
        return p.coefsC


def flat_coefs(p):
    cp = _coefs(p)
    return cp.reshape(numpy.prod(p.shape), cp.shape[-1])


def polynomial_size(p):
    return p.polynomial_size


def prepare_ifft_input_(rev_in, a, coeff, N):
    rev_in[:,:N] = a * coeff
    rev_in[:,N:] = -rev_in[:,:N]


def prepare_ifft_output_(res, rev_out, N):
    # FIXME: when Julia is smart enough, can be replaced by:
    res[:,:N//2] = rev_out[:,1:N+1:2]


def ip_ifft_(result: LagrangeHalfCPolynomialArray, p: IntPolynomialArray):
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], 2 * N), numpy.float64)
    prepare_ifft_input_(in_arr, a, 1/2, N)
    out_arr = numpy.fft.rfft(in_arr)
    prepare_ifft_output_(res, out_arr, N)


def tp_ifft_(result: LagrangeHalfCPolynomialArray, p: TorusPolynomialArray):
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], 2 * N), numpy.float64)
    prepare_ifft_input_(in_arr, a, 1 / 2**33, N)
    out_arr = numpy.fft.rfft(in_arr)
    prepare_ifft_output_(res, out_arr, N)


def prepare_fft_input_(fw_in, a, N):
    fw_in[:,0:N+1:2] = 0
    fw_in[:,1:N+1:2] = a


def prepare_fft_output_(res, fw_out, coeff, N):
    res[:,:] = float_to_int32(fw_out[:,:N] * coeff)


def tp_fft_(result: TorusPolynomialArray, p: LagrangeHalfCPolynomialArray):
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], N + 1), numpy.complex128)
    prepare_fft_input_(in_arr, a, N)
    out_arr = numpy.fft.irfft(in_arr)

    # the first part is from the original libtfhe;
    # the second part is from a different FFT scaling in Julia
    coeff = (2**32 / N) * (2 * N)
    prepare_fft_output_(res, out_arr, coeff, N)


def tp_add_mul_(
        result: TorusPolynomialArray, poly1: IntPolynomialArray, poly2: TorusPolynomialArray):

    N = polynomial_size(result)
    tmp1 = LagrangeHalfCPolynomialArray(N, poly1.shape)
    tmp2 = LagrangeHalfCPolynomialArray(N, poly2.shape)
    tmp3 = LagrangeHalfCPolynomialArray(N, result.shape)
    tmpr = TorusPolynomialArray(N, result.shape)
    ip_ifft_(tmp1, poly1)
    tp_ifft_(tmp2, poly2)
    lp_mul_(tmp3, tmp1, tmp2)
    tp_fft_(tmpr, tmp3)
    tp_add_to_(result, tmpr)


#MISC OPERATIONS

# sets to zero
def lp_clear_(reps: LagrangeHalfCPolynomialArray):
    reps.coefsC.fill(0)


# termwise multiplication in Lagrange space */
def lp_mul_(
        result: LagrangeHalfCPolynomialArray,
        a: LagrangeHalfCPolynomialArray,
        b: LagrangeHalfCPolynomialArray):

    numpy.copyto(result.coefsC, a.coefsC * b.coefsC)


# Torus polynomial functions

# TorusPolynomial = 0
def tp_clear_(result: TorusPolynomialArray):
    result.coefsT.fill(0)


# TorusPolynomial += TorusPolynomial
def tp_add_to_(result: TorusPolynomialArray, poly2: TorusPolynomialArray):
    result.coefsT += poly2.coefsT


# result = (X^ai-1) * source
def tp_mul_by_xai_minus_one_(out: TorusPolynomialArray, ais, in_: TorusPolynomialArray):
    out_c = out.coefsT
    in_c = in_.coefsT

    N = out_c.shape[-1]
    for i in range(out.shape[0]):
        ai = ais[i]
        if ai < N:
            out_c[i,:,:ai] = -in_c[i,:,(N-ai):N] - in_c[i,:,:ai] # sur que i-a<0
            out_c[i,:,ai:N] = in_c[i,:,:(N-ai)] - in_c[i,:,ai:N] # sur que N>i-a>=0
        else:
            aa = ai - N
            out_c[i,:,:aa] = in_c[i,:,(N-aa):N] - in_c[i,:,:aa] # sur que i-a<0
            out_c[i,:,aa:N] = -in_c[i,:,:(N-aa)] - in_c[i,:,aa:N] # sur que N>i-a>=0


# result= X^{a}*source
def tp_mul_by_xai_(out, ais, in_):
    out_c = out.coefsT
    in_c = in_.coefsT

    N = out_c.shape[-1]
    for i in range(out.shape[0]):
        ai = ais[i]
        if ai < N:
            out_c[i,:ai] = -in_c[i,(N-ai):N] # sur que i-a<0
            out_c[i,ai:N] = in_c[i,:(N-ai)] # sur que N>i-a>=0
        else:
            aa = ai - N
            out_c[i,:aa] = in_c[i,(N-aa):N] # sur que i-a<0
            out_c[i,aa:N] = -in_c[i,:(N-aa)] # sur que N>i-a>=0
