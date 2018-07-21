import numpy

from .gpu_polynomials import LagrangeHalfCPolynomialArray, TorusPolynomialArray
from .polynomial_transform import (
    forward_transform_ref,
    inverse_transform_ref,
    transformed_space_mul_ref,
    )


# create an homogeneous tlwe sample
def TLweSymEncryptZero_ref(shape, alpha: float, params: 'TLweParams'):
    N = params.N
    k = params.k

    def _kernel(result_a, result_cv, key, noises1, noises2):

        tmp1 = LagrangeHalfCPolynomialArray(None, N, (k,))
        tmp2 = LagrangeHalfCPolynomialArray(None, N, shape + (k,))
        tmp3 = LagrangeHalfCPolynomialArray(None, N, shape + (k,))
        tmpr = TorusPolynomialArray(None, N, shape + (k,))

        tmp1.coefsC = forward_transform_ref(key)
        tmp2.coefsC = forward_transform_ref(noises1)
        numpy.copyto(tmp3.coefsC, transformed_space_mul_ref(tmp1.coefsC, tmp2.coefsC))
        tmpr.coefsT = inverse_transform_ref(tmp3.coefsC)

        result_a[:,:,:,:k,:] = noises1
        result_a[:,:,:,k,:] = noises2
        for i in range(k):
            result_a[:,:,:,k,:] += tmpr.coefsT[:,:,:,i,:]

        result_cv.fill(alpha**2)

    return _kernel


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
