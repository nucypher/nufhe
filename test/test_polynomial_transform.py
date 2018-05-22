import numpy

from tfhe.polynomial_transform import (
    ForwardTransformFFT, InverseTransformFFT, ForwardTransform, InverseTransform)

from reikna.helpers import product


def forward_transform_fft_ref(a):
    batch_shape = a.shape[:-1]
    batch_size = product(batch_shape)
    N = a.shape[-1]

    idx = numpy.arange(N // 2)
    coeff = numpy.exp(-2j * numpy.pi * idx / 2 / N)

    flat_a = a.reshape(batch_size, N)
    input_arr = (flat_a[:,:N//2] - 1j * flat_a[:,N//2:]) * coeff
    output_arr = numpy.fft.fft(input_arr)
    output_shape = batch_shape + (N // 2,)

    return output_arr.reshape(output_shape)


def inverse_transform_fft_ref(a):
    batch_shape = a.shape[:-1]
    batch_size = product(batch_shape)
    N = a.shape[-1] * 2

    idx = numpy.arange(N // 2)
    coeff = numpy.exp(-2j * numpy.pi * idx / 2 / N)

    flat_a = a.reshape(batch_size, N // 2)
    output_arr = numpy.fft.ifft(flat_a).conj() * coeff
    output_shape = batch_shape + (N,)

    return numpy.concatenate([output_arr.real, output_arr.imag], axis=1).reshape(output_shape)


def test_forward_transform_fft(thread):

    batch_shape = (10,)
    N = 1024

    fft = ForwardTransformFFT(batch_shape, N)
    fftc = fft.compile(thread)

    a = numpy.random.normal(size=fft.parameter.input.shape).astype(fft.parameter.input.dtype)

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(fft.parameter.output)
    fftc(res_dev, a_dev)

    res_test = res_dev.get()
    res_ref = forward_transform_fft_ref(a)

    assert numpy.allclose(res_test, res_ref)


def test_inverse_transform_fft(thread):

    batch_shape = (10,)
    N = 1024

    ifft = InverseTransformFFT(batch_shape, N)
    ifftc = ifft.compile(thread)

    ishape = ifft.parameter.input.shape
    idtype = ifft.parameter.input.dtype

    a = (numpy.random.normal(size=ishape).astype(idtype)
        + 1j * numpy.random.normal(size=ishape).astype(idtype))

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(ifft.parameter.output)
    ifftc(res_dev, a_dev)

    res_test = res_dev.get()
    res_ref = inverse_transform_fft_ref(a)

    assert numpy.allclose(res_test, res_ref)


def forward_transform_ref(a):
    return forward_transform_fft_ref(a.astype(numpy.float64))


def inverse_transform_ref(a):
    return numpy.round(inverse_transform_fft_ref(a)).astype(numpy.int64).astype(numpy.int32)


def test_forward_transform(thread):

    batch_shape = (10,)
    N = 1024

    ft = ForwardTransform(batch_shape, N)
    ftc = ft.compile(thread)

    a = numpy.random.randint(
        -2**31, 2**31, size=ft.parameter.input.shape, dtype=ft.parameter.input.dtype)

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(ftc.parameter.output)
    ftc(res_dev, a_dev)

    res_test = res_dev.get()
    res_ref = forward_transform_ref(a)

    assert numpy.allclose(res_test, res_ref)


def test_inverse_transform(thread):

    batch_shape = (10,)
    N = 1024

    ift = InverseTransform(batch_shape, N)
    iftc = ift.compile(thread)

    ishape = ift.parameter.input.shape
    idtype = ift.parameter.input.dtype

    # Using a large scale so that the result is much greater than 1
    # and does not turn into all 0 after rounding off.
    a = (numpy.random.normal(size=ishape, scale=N * 1000).astype(idtype)
        + 1j * numpy.random.normal(size=ishape, scale=N * 1000).astype(idtype))

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(ift.parameter.output)
    iftc(res_dev, a_dev)

    res_test = res_dev.get()
    res_ref = inverse_transform_ref(a)

    assert numpy.allclose(res_test, res_ref)


def poly_mul_ref(p1, p2):
    N = p1.shape[-1]

    result = numpy.empty_like(p1)

    for i in range(N):
        result[:,i] = (p1[:,:i+1] * p2[:,i::-1]).sum(1) - (p1[:,i+1:] * p2[:,:i:-1]).sum(1)

    return result


def test_polynomial_multiplication(thread):

    batch_shape = (10,)
    N = 1024

    ft = ForwardTransform(batch_shape, N)
    ift = InverseTransform(batch_shape, N)

    ftc = ft.compile(thread)
    iftc = ift.compile(thread)

    a = numpy.random.randint(
        -2**31, 2**31, size=ft.parameter.input.shape, dtype=ft.parameter.input.dtype)
    b = numpy.random.randint(
        -1000, 1000, size=ft.parameter.input.shape, dtype=ft.parameter.input.dtype)

    a_dev = thread.to_device(a)
    b_dev = thread.to_device(b)
    a_tr_dev = thread.empty_like(ftc.parameter.output)
    b_tr_dev = thread.empty_like(ftc.parameter.output)
    res_dev = thread.empty_like(iftc.parameter.output)

    ftc(a_tr_dev, a_dev)
    ftc(b_tr_dev, b_dev)
    res_tr = a_tr_dev.get() * b_tr_dev.get()
    res_tr_dev = thread.to_device(res_tr)
    iftc(res_dev, res_tr_dev)

    res_test = res_dev.get()
    res_ref = poly_mul_ref(a, b)

    assert numpy.allclose(res_test, res_ref)
