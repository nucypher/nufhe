import numpy

from tfhe.numeric_functions import Torus32, Float
from tfhe.tlwe import TLweParams
from tfhe.gpu_tlwe import (
    TLweNoiselessTrivial, TLweExtractLweSample,
    TLweSymEncryptZero, TLweSymEncryptZero_ref)

import tfhe.random_numbers as rn


def int_prod(arr):
    return numpy.prod(arr, dtype=numpy.int32)


def tLweNoiselessTrivial_reference(result_a, result_current_variances, mu):
    assert len(result_a.shape) == 3
    assert result_current_variances.shape == result_a.shape[:-2]
    assert mu.shape == (result_a.shape[0], result_a.shape[-1])
    assert result_a.dtype == mu.dtype

    k = result_a.shape[1] - 1
    result_a[:,:k,:] = 0
    result_a[:,k,:] = mu
    result_current_variances.fill(0.)



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



def tLweExtractLweSample_reference(result_a, result_b, tlwe_a):

    N = tlwe_a.shape[-1]
    k = tlwe_a.shape[-2] - 1
    assert result_a.shape[-1] == k*N
    assert result_a.shape[:-1] == tlwe_a.shape[:-2]
    assert result_b.shape == tlwe_a.shape[:-2]
    assert result_a.dtype == tlwe_a.dtype
    assert result_b.dtype == tlwe_a.dtype

    batch = int_prod(tlwe_a.shape[:-2])

    a_view = result_a.reshape(batch, k, N)
    b_view = result_b.reshape(batch)
    tlwe_a_view = tlwe_a.reshape(batch, k + 1, N)

    a_view[:,:,0] = tlwe_a_view[:, :k, 0]
    a_view[:,:,1:] = -tlwe_a_view[:, :k, :0:-1]

    numpy.copyto(b_view, tlwe_a_view[:, k, 0])


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

    k = 1
    l = 2
    N = 1024
    alpha = 5e-9
    params = TLweParams(N, k, alpha, alpha)
    shape = (5, k + 1, l)

    result_a = numpy.empty(shape + (k + 1, N), numpy.int32)
    result_cv = numpy.empty(shape, numpy.float64)
    noises2 = rn._rand_gaussian_torus32(rng, 0, alpha, shape + (N,))
    noises1 = rn._rand_uniform_torus32(rng, shape + (k, N))
    key = rn._rand_uniform_int32(rng, (k, N))

    comp = TLweSymEncryptZero(shape, alpha, params).compile(thread)
    ref = TLweSymEncryptZero_ref(shape, alpha, params)

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

