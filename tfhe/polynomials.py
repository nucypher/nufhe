import numpy

from .numeric_functions import *

from .polynomial_transform import (
    ForwardTransform, InverseTransform, transformed_dtype, transformed_length)

global_thread = None


def get_global_thread():
    return global_thread


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

    def to_gpu(self, thr):
        self.coefsT = thr.to_device(self.coefsT)

    def from_gpu(self):
        self.coefsT = self.coefsT.get()


# This structure is used for FFT operations, and is a representation
# over C of a polynomial in R[X]/X^N+1
class LagrangeHalfCPolynomialArray:
    def __init__(self, N, shape):
        assert N % 2 == 0
        self.coefsC = numpy.empty(shape + (transformed_length(N),), transformed_dtype())
        self.polynomial_size = N
        self.shape = shape

    def to_gpu(self, thr):
        self.coefsC = thr.to_device(self.coefsC.astype(transformed_dtype()))

    def from_gpu(self):
        self.coefsC = self.coefsC.get().astype(transformed_dtype())


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
    a = numpy.ascontiguousarray(flat_coefs(p))
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], 2 * N), Float)
    prepare_ifft_input_(in_arr, a, 1/2, N)
    out_arr = numpy.fft.rfft(in_arr)
    prepare_ifft_output_(res, out_arr, N)


def ip_ifft_transformed(result: LagrangeHalfCPolynomialArray, p: IntPolynomialArray):
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    idxs = numpy.arange(N//2)
    in_arr = (a[:,:N//2] - 1j * a[:,N//2:]) * numpy.exp(-2j * numpy.pi * idxs / N / 2)
    out_arr = numpy.fft.fft(in_arr)
    numpy.copyto(res, out_arr)


FFT_COEFF = 2**33


def tp_ifft_(result: LagrangeHalfCPolynomialArray, p: TorusPolynomialArray):
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], 2 * N), Float)
    prepare_ifft_input_(in_arr, a, 1 / FFT_COEFF, N)
    out_arr = numpy.fft.rfft(in_arr)
    prepare_ifft_output_(res, out_arr, N)


def tp_ifft_transformed(result: LagrangeHalfCPolynomialArray, p: TorusPolynomialArray):
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    idxs = numpy.arange(N//2)
    in_arr = (a[:,:N//2] - 1j * a[:,N//2:]) * numpy.exp(-2j * numpy.pi * idxs / N / 2)
    out_arr = numpy.fft.fft(in_arr)
    numpy.copyto(res, out_arr)


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
    coeff = FFT_COEFF
    prepare_fft_output_(res, out_arr, coeff, N)


def tp_fft_transformed(result: TorusPolynomialArray, p: LagrangeHalfCPolynomialArray):
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    out_arr = numpy.fft.ifft(a)
    idxs = numpy.arange(N//2)
    out_arr = out_arr.conj() * numpy.exp(-2j * numpy.pi * idxs / N / 2)
    numpy.copyto(res, numpy.concatenate([
        float_to_int32(out_arr.real),
        float_to_int32(out_arr.imag)], axis=1))


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
