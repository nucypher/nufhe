import time

import numpy

from tfhe.numeric_functions import Torus32
from tfhe.polynomials_gpu import ShiftTorusPolynomial
from tfhe.polynomials_cpu import ShiftTorusPolynomial_ref


def test_mul_by_xai(thread):

    N = 16

    data = numpy.random.randint(0, 10000, size=(300, 10, N))
    ais = numpy.random.randint(0, 2 * N, size=(300, 10))
    res_ref = numpy.empty_like(data)

    data_dev = thread.to_device(data)
    ais_dev = thread.to_device(ais)
    res_dev = thread.empty_like(res_ref)

    comp = ShiftTorusPolynomial(ais, data, minus_one=False).compile(thread)
    ref = ShiftTorusPolynomial_ref(ais, data, minus_one=False)

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

    comp = ShiftTorusPolynomial(ais, data, minus_one=False, invert_ais=True).compile(thread)
    ref = ShiftTorusPolynomial_ref(ais, data, minus_one=False, invert_ais=True)

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

    comp = ShiftTorusPolynomial(ais, data, minus_one=True, ai_view=True).compile(thread)
    ref = ShiftTorusPolynomial_ref(ais, data, minus_one=True, ai_view=True)

    comp(res_dev, ais_dev, 3, data_dev)
    res_test = res_dev.get()

    ref(res_ref, ais, 3, data)

    assert numpy.allclose(res_test, res_ref)
