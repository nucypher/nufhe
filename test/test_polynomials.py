import numpy

from tfhe.numeric_functions import Torus32, Complex
from tfhe.gpu_polynomials import I2C_FFT, C2I_FFT, TPMulByXai


def int_prod(arr):
    return numpy.prod(arr, dtype=numpy.int32)


# coeff=2 for Int polynomial, coeff=2^33 for Torus polynomial
def ip_ifft_reference(res, p, coeff):

    N = p.shape[-1]

    assert res.shape[-1] == N // 2
    assert res.shape[:-1] == p.shape[:-1]
    assert p.dtype in (Torus32, numpy.int32)
    assert res.dtype == Complex

    batch = int_prod(p.shape[:-1])
    p = p.reshape(batch, N)
    res = res.reshape(batch, N // 2)

    in_arr = numpy.empty((batch, 2 * N), numpy.float64)

    in_arr[:,:N] = p / coeff
    in_arr[:,N:] = -in_arr[:,:N]

    out_arr = numpy.fft.rfft(in_arr)

    res[:,:N//2] = out_arr[:,1:N+1:2]


def tp_fft_reference(res, p):

    N = p.shape[-1] * 2

    assert res.shape[-1] == N
    assert res.shape[:-1] == p.shape[:-1]
    assert p.dtype == Complex
    assert res.dtype == Torus32

    batch = int_prod(p.shape[:-1])
    p = p.reshape(batch, N // 2)
    res = res.reshape(batch, N)

    in_arr = numpy.empty((batch, N + 1), Complex)
    in_arr[:,0:N+1:2] = 0
    in_arr[:,1:N+1:2] = p

    out_arr = numpy.fft.irfft(in_arr)

    # the first part is from the original libtfhe;
    # the second part is from a different FFT scaling in Julia
    coeff = (2**32 / N) * (2 * N)

    res[:,:] = numpy.round(out_arr[:,:N] * coeff).astype(numpy.int64).astype(numpy.int32)


# minus_one=True: result = (X^ai-1) * source
# minus_one=False: result = X^{a} * source
def tp_mul_by_xai_reference(out_c, ais, in_c, minus_one=False):
    assert out_c.dtype == in_c.dtype
    assert out_c.shape == in_c.shape
    assert ais.shape == (out_c.shape[0],)

    bc_shape = (out_c.shape[0], int_prod(out_c.shape[1:-1]), out_c.shape[-1])
    out_c = out_c.reshape(bc_shape)
    in_c = in_c.reshape(bc_shape)

    N = out_c.shape[-1]
    for i in range(out_c.shape[0]):
        ai = ais[i]
        if ai < N:
            out_c[i,:,:ai] = -in_c[i,:,(N-ai):N] - (in_c[i,:,:ai] if minus_one else 0)
            out_c[i,:,ai:N] = in_c[i,:,:(N-ai)] - (in_c[i,:,ai:N] if minus_one else 0)
        else:
            aa = ai - N
            out_c[i,:,:aa] = in_c[i,:,(N-aa):N] - (in_c[i,:,:aa] if minus_one else 0)
            out_c[i,:,aa:N] = -in_c[i,:,:(N-aa)] - (in_c[i,:,aa:N] if minus_one else 0)


def test_mul_by_xai(thread):

    N = 16

    data = numpy.random.randint(0, 10000, size=(300, 10, N))
    ais = numpy.random.randint(0, 2 * N, size=300)
    res_ref = numpy.empty_like(data)

    data_dev = thread.to_device(data)
    ais_dev = thread.to_device(ais)
    res_dev = thread.empty_like(res_ref)

    comp = TPMulByXai(ais, data, minus_one=False).compile(thread)

    comp(res_dev, ais_dev, data_dev)
    res_test = res_dev.get()

    tp_mul_by_xai_reference(res_ref, ais, data, minus_one=False)

    assert numpy.allclose(res_test, res_ref)


def test_mul_by_xai_invert_ais(thread):

    N = 16

    data = numpy.random.randint(0, 10000, size=(300, 10, N))
    ais = numpy.random.randint(0, 2 * N, size=300)
    res_ref = numpy.empty_like(data)

    data_dev = thread.to_device(data)
    ais_dev = thread.to_device(ais)
    res_dev = thread.empty_like(res_ref)

    comp = TPMulByXai(ais, data, minus_one=False, invert_ais=True).compile(thread)

    comp(res_dev, ais_dev, data_dev)
    res_test = res_dev.get()

    tp_mul_by_xai_reference(res_ref, 2 * N - ais, data, minus_one=False)

    assert numpy.allclose(res_test, res_ref)


def test_mul_by_xai_minus_one(thread):

    N = 16

    data = numpy.random.randint(0, 10000, size=(300, 10, N))
    ais = numpy.random.randint(0, 2 * N, size=300)
    res_ref = numpy.empty_like(data)

    data_dev = thread.to_device(data)
    ais_dev = thread.to_device(ais)
    res_dev = thread.empty_like(res_ref)

    comp = TPMulByXai(ais, data, minus_one=True).compile(thread)

    comp(res_dev, ais_dev, data_dev)
    res_test = res_dev.get()

    tp_mul_by_xai_reference(res_ref, ais, data, minus_one=True)

    assert numpy.allclose(res_test, res_ref)


def test_i2c_fft(thread):

    N = 1024
    batch_shape = (500, 2, 2)

    data = numpy.random.randint(-2**31, 2**31, size=batch_shape + (N,), dtype=numpy.int32)
    res_ref = numpy.empty(batch_shape + (N//2,), dtype=Complex)

    data_dev = thread.to_device(data)
    res_dev = thread.empty_like(res_ref)

    ipfft = I2C_FFT(data, 2).compile(thread)

    ipfft(res_dev, data_dev)
    res_test = res_dev.get()

    ip_ifft_reference(res_ref, data, 2)

    assert numpy.allclose(res_test, res_ref)


def test_c2i_fft(thread):

    N = 1024
    batch_shape = (500, 2, 2)

    data = (
        numpy.random.normal(size=batch_shape + (N//2,))
        + 1j * numpy.random.normal(size=batch_shape + (N//2,)))
    res_ref = numpy.empty(batch_shape + (N,), Torus32)

    data_dev = thread.to_device(data)
    res_dev = thread.empty_like(res_ref)

    pfft = C2I_FFT(data).compile(thread)

    pfft(res_dev, data_dev)
    res_test = res_dev.get()

    tp_fft_reference(res_ref, data)

    assert numpy.allclose(res_test, res_ref)
