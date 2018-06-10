import numpy

import tfhe.polynomial_transform
from tfhe.polynomial_transform import (
    forward_transform_ref, inverse_transform_ref, transformed_space_mul_ref,
    ForwardTransform, InverseTransform)

from reikna.helpers import product


def test_forward_transform(thread):

    batch_shape = (10,)
    N = 1024

    a = numpy.random.randint(-2**31, 2**31, size=batch_shape+(N,), dtype=numpy.int32)

    fft = ForwardTransform(batch_shape, N)
    fftc = fft.compile(thread)

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(fft.parameter.output)
    fftc(res_dev, a_dev)

    res_test = res_dev.get()

    res_ref = forward_transform_ref(a)

    assert res_test.dtype == res_ref.dtype
    assert numpy.allclose(res_test, res_ref)


def test_inverse_transform(thread):

    batch_shape = (10,)
    N = 1024

    a = numpy.random.randint(-2**31, 2**31, size=batch_shape+(N,), dtype=numpy.int32)
    tr_a = forward_transform_ref(a)

    ifft = InverseTransform(batch_shape, N)
    ifftc = ifft.compile(thread)

    tr_a_dev = thread.to_device(tr_a)
    res_dev = thread.empty_like(ifft.parameter.output)
    ifftc(res_dev, tr_a_dev)

    res_test = res_dev.get()
    res_ref = inverse_transform_ref(tr_a)

    assert res_test.dtype == res_ref.dtype
    assert res_test.dtype == numpy.int32
    assert numpy.allclose(res_test, res_ref)
    assert numpy.allclose(res_test, a)


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
    res_tr = transformed_space_mul_ref(a_tr_dev.get(), b_tr_dev.get())
    res_tr_dev = thread.to_device(res_tr)
    iftc(res_dev, res_tr_dev)

    res_test = res_dev.get()
    res_ref = poly_mul_ref(a, b)

    assert numpy.allclose(res_test, res_ref)
