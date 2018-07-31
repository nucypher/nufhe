import time

import numpy

from tfhe.numeric_functions import Torus32
from tfhe.gpu_polynomials import TPMulByXai


def int_prod(arr):
    return numpy.prod(arr, dtype=numpy.int32)


# minus_one=True: result = (X^ai-1) * source
# minus_one=False: result = X^{a} * source
def TPMulByXai_ref(ais, arr, ai_view=False, minus_one=False, invert_ais=False):

    if ai_view:
        assert len(ais.shape) == len(arr.shape) - 1
        assert ais.shape[:-1] == arr.shape[:-2]
    else:
        assert len(ais.shape) == len(arr.shape) - 1
        assert ais.shape == arr.shape[:-1]

    N = arr.shape[-1]

    def _kernel(output, ais, ai_idx, input_):

        if ai_view:
            ai_batch_len = int_prod(ais.shape[:-int(ai_view)])
            arr_batch_len = int_prod(arr.shape[len(ais.shape)-1:-1])

            ais = ais.reshape(ai_batch_len, ais.shape[-1])
            ais = ais[:,ai_idx]

            out_c = output.reshape(ai_batch_len, arr_batch_len, N)
            in_c = input_.reshape(ai_batch_len, arr_batch_len, N)

        else:
            ais = ais.flatten()
            ai_batch_len = ais.size

            out_c = output.reshape(ai_batch_len, 1, N)
            in_c = input_.reshape(ai_batch_len, 1, N)


        if invert_ais:
            ais = 2 * N - ais

        for i in range(ai_batch_len):
            ai = ais[i]
            if ai < N:
                out_c[i,:,:ai] = -in_c[i,:,(N-ai):N]
                out_c[i,:,ai:N] = in_c[i,:,:(N-ai)]
            else:
                aa = ai - N
                out_c[i,:,:aa] = in_c[i,:,(N-aa):N]
                out_c[i,:,aa:N] = -in_c[i,:,:(N-aa)]

        if minus_one:
            out_c -= in_c

    return _kernel


def test_mul_by_xai(thread):

    N = 16

    data = numpy.random.randint(0, 10000, size=(300, 10, N))
    ais = numpy.random.randint(0, 2 * N, size=(300, 10))
    res_ref = numpy.empty_like(data)

    data_dev = thread.to_device(data)
    ais_dev = thread.to_device(ais)
    res_dev = thread.empty_like(res_ref)

    comp = TPMulByXai(ais, data, minus_one=False).compile(thread)
    ref = TPMulByXai_ref(ais, data, minus_one=False)

    comp(res_dev, ais_dev, 0, data_dev)
    res_test = res_dev.get()

    ref(res_ref, ais, 0, data)

    assert numpy.allclose(res_test, res_ref)


def test_mul_by_xai_invert_ais(thread):

    N = 16

    data = numpy.random.randint(0, 10000, size=(300, 10, N))
    ais = numpy.random.randint(0, 2 * N, size=(300, 10))
    res_ref = numpy.empty_like(data)

    data_dev = thread.to_device(data)
    ais_dev = thread.to_device(ais)
    res_dev = thread.empty_like(res_ref)

    comp = TPMulByXai(ais, data, minus_one=False, invert_ais=True).compile(thread)
    ref = TPMulByXai_ref(ais, data, minus_one=False, invert_ais=True)

    comp(res_dev, ais_dev, 0, data_dev)
    res_test = res_dev.get()

    ref(res_ref, ais, 0, data)

    assert numpy.allclose(res_test, res_ref)


def test_mul_by_xai_minus_one(thread):

    N = 16

    data = numpy.random.randint(0, 10000, size=(20, 10, 2, N))
    ais = numpy.random.randint(0, 2 * N, size=(20, 10, 15))
    res_ref = numpy.empty_like(data)

    data_dev = thread.to_device(data)
    ais_dev = thread.to_device(ais)
    res_dev = thread.empty_like(res_ref)

    comp = TPMulByXai(ais, data, minus_one=True, ai_view=True).compile(thread)
    ref = TPMulByXai_ref(ais, data, minus_one=True, ai_view=True)

    comp(res_dev, ais_dev, 3, data_dev)
    res_test = res_dev.get()

    ref(res_ref, ais, 3, data)

    assert numpy.allclose(res_test, res_ref)
