import numpy

from tfhe.numeric_functions import Torus32
from tfhe.gpu_tlwe import TLweNoiselessTrivial, TLweExtractLweSample


def int_prod(arr):
    return numpy.prod(arr, dtype=numpy.int32)


def tLweNoiselessTrivial_reference(result_a, result_current_variances, mu):
    assert len(result_a.shape) == 3
    assert result_current_variances.shape == result_a.shape[:-1]
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
    cv_ref = numpy.empty((l, k+1), numpy.float64)

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
