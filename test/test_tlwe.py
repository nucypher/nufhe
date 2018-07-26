import numpy

from tfhe.keys import TFHEParameters
from tfhe.numeric_functions import Torus32, Float
from tfhe.tlwe import TLweParams
from tfhe.tlwe_gpu import (
    TLweNoiselessTrivial,
    TLweExtractLweSample,
    TLweSymEncryptZero,
    )
from tfhe.tlwe_cpu import (
    TLweSymEncryptZero_ref,
    tLweNoiselessTrivial_reference,
    tLweExtractLweSample_reference,
    )
from tfhe.performance import performance_parameters
import tfhe.random_numbers as rn


def test_tLweNoiselessTrivial(thread):

    l = 10
    k = 2
    n = 500

    mu = numpy.random.randint(-2**31, 2**31, size=(l, n), dtype=Torus32)
    a_ref = numpy.empty((l, k+1, n), Torus32)
    cv_ref = numpy.empty((l,), Float)

    a_dev = thread.to_device(a_ref)
    cv_dev = thread.empty_like(cv_ref)
    mu_dev = thread.to_device(mu)

    comp = TLweNoiselessTrivial(a_ref).compile(thread)

    comp(a_dev, cv_dev, mu_dev)
    a_test = a_dev.get()
    cv_test = cv_dev.get()

    tLweNoiselessTrivial_reference(a_ref, cv_ref, mu)

    assert numpy.allclose(a_test, a_ref)
    assert numpy.allclose(cv_test, cv_ref)


def test_tLweExtractLweSample(thread):

    ml = (2, 5)
    k = 2
    N = 1024
    tlwe_a = numpy.random.randint(-2**31, 2**31, size=ml + (k + 1, N), dtype=Torus32)

    a_ref = numpy.empty(ml + (k * N,), Torus32)
    b_ref = numpy.empty(ml, Torus32)

    tlwe_a_dev = thread.to_device(tlwe_a)
    a_dev = thread.empty_like(a_ref)
    b_dev = thread.empty_like(b_ref)

    comp = TLweExtractLweSample(tlwe_a).compile(thread)

    comp(a_dev, b_dev, tlwe_a_dev)
    a_test = a_dev.get()
    b_test = b_dev.get()

    tLweExtractLweSample_reference(a_ref, b_ref, tlwe_a)

    assert numpy.allclose(a_test, a_ref)
    assert numpy.allclose(b_test, b_ref)


def test_TLweSymEncryptZero(thread):

    rng = numpy.random.RandomState(123)

    tfhe_params = TFHEParameters()
    perf_params = performance_parameters()
    params = tfhe_params.tgsw_params.tlwe_params

    k = params.mask_size
    l = tfhe_params.tgsw_params.decomp_length
    N = params.polynomial_degree
    alpha = params.alpha_min

    shape = (5, k + 1, l)

    result_a = numpy.empty(shape + (k + 1, N), numpy.int32)
    result_cv = numpy.empty(shape, numpy.float64)
    noises2 = rn._rand_gaussian_torus32(rng, 0, alpha, shape + (N,))
    noises1 = rn._rand_uniform_torus32(rng, shape + (k, N))
    key = rn._rand_uniform_int32(rng, (k, N))

    comp = TLweSymEncryptZero(shape, alpha, params, perf_params).compile(thread)
    ref = TLweSymEncryptZero_ref(shape, alpha, params, perf_params)

    result_a_dev = thread.empty_like(result_a)
    result_cv_dev = thread.empty_like(result_cv)
    noises1_dev = thread.to_device(noises1)
    noises2_dev = thread.to_device(noises2)
    key_dev = thread.to_device(key)

    comp(result_a_dev, result_cv_dev, key_dev, noises1_dev, noises2_dev)
    ref(result_a, result_cv, key, noises1, noises2)

    result_a_test = result_a_dev.get()
    result_cv_test = result_cv_dev.get()

    assert (result_a_test == result_a).all()
    assert numpy.allclose(result_cv_test, result_cv)

