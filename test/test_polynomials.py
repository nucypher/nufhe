import time

import numpy

from tfhe.numeric_functions import Torus32
from tfhe.gpu_polynomials import TPMulByXai


def int_prod(arr):
    return numpy.prod(arr, dtype=numpy.int32)


# minus_one=True: result = (X^ai-1) * source
# minus_one=False: result = X^{a} * source
def tp_mul_by_xai_reference(
        out_c, ais, ai_idx, in_c, ai_view=False, invert_ais=False, minus_one=False):

    assert out_c.dtype == in_c.dtype
    assert out_c.shape == in_c.shape
    assert ais.shape == (out_c.shape[0],)

    bc_shape = (out_c.shape[0], int_prod(out_c.shape[1:-1]), out_c.shape[-1])
    out_c = out_c.reshape(bc_shape)
    in_c = in_c.reshape(bc_shape)

    N = out_c.shape[-1]

    if ai_view:
        ais = ais[:,ai_idx]

    if invert_ais:
        ais = 2 * N - ais

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

    comp(res_dev, ais_dev, 0, data_dev)
    res_test = res_dev.get()

    tp_mul_by_xai_reference(res_ref, ais, 0, data, minus_one=False)

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

    comp(res_dev, ais_dev, 0, data_dev)
    res_test = res_dev.get()

    tp_mul_by_xai_reference(res_ref, ais, 0, data, invert_ais=True, minus_one=False)

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

    comp(res_dev, ais_dev, 0, data_dev)
    res_test = res_dev.get()

    tp_mul_by_xai_reference(res_ref, ais, 0, data, minus_one=True)

    assert numpy.allclose(res_test, res_ref)
