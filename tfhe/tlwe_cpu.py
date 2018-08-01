import numpy

from .gpu_polynomials import LagrangeHalfCPolynomialArray, TorusPolynomialArray
from .polynomial_transform import get_transform


# create an homogeneous tlwe sample
def TLweSymEncryptZero_ref(shape, noise: float, params: 'TLweParams', perf_params):
    N = params.polynomial_degree
    k = params.mask_size

    transform_type = params.transform_type
    transform = get_transform(transform_type)

    def _kernel(result_a, result_cv, key, noises1, noises2):

        tmp1 = LagrangeHalfCPolynomialArray(None, transform_type, N, (k,))
        tmp2 = LagrangeHalfCPolynomialArray(None, transform_type, N, shape + (k,))
        tmp3 = LagrangeHalfCPolynomialArray(None, transform_type, N, shape + (k,))
        tmpr = TorusPolynomialArray(None, N, shape + (k,))

        tmp1.coefsC = transform.forward_transform_ref(key)
        tmp2.coefsC = transform.forward_transform_ref(noises1)
        numpy.copyto(tmp3.coefsC, transform.transformed_space_mul_ref(tmp1.coefsC, tmp2.coefsC))
        tmpr.coefsT = transform.inverse_transform_ref(tmp3.coefsC)

        result_a[:,:,:,:k,:] = noises1
        result_a[:,:,:,k,:] = noises2
        for i in range(k):
            result_a[:,:,:,k,:] += tmpr.coefsT[:,:,:,i,:]

        result_cv.fill(noise**2)

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
