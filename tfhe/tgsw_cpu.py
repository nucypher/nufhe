import numpy

from tfhe.numeric_functions import Torus32
from tfhe.polynomial_transform import get_transform


def TGswTorus32PolynomialDecompH_reference(_result, params: 'TGswParams'):

    def _kernel(result, sample):

        N = result.shape[-1]
        l = result.shape[-2]
        k = result.shape[-3] - 1
        batch = result.shape[:-3]

        assert sample.shape == batch + (k + 1, N)
        assert result.dtype == numpy.int32
        assert sample.dtype == Torus32

        bs_log2_base = params.bs_log2_base
        base = 2**bs_log2_base
        maskMod = base - 1
        halfBg = 2**(bs_log2_base - 1)
        offset = params.offset

        decal = lambda p: 32 - p * bs_log2_base

        ps = numpy.arange(1, l+1).reshape((1,) * len(batch) + (1, l, 1))
        sample_coefs = sample.reshape(sample.shape[:-1] + (1, N))

        # do the decomposition
        numpy.copyto(result, (((sample_coefs + offset) >> decal(ps)) & maskMod) - halfBg)

    return _kernel


def TLweFFTAddMulRTo_reference(res, gsw, tlwe_params):

    transform = get_transform(tlwe_params.transform_type)

    def _kernel(res, decaFFT, gsw, bk_idx):

        batch_shape = res.shape[:-2]
        k = res.shape[-2] - 1
        transformed_N = res.shape[-1]
        l = decaFFT.shape[-2]

        assert decaFFT.shape == batch_shape + (k + 1, l, transformed_N)
        assert gsw.shape[-4:] == (k + 1, l, k + 1, transformed_N)
        assert res.shape == batch_shape + (k + 1, transformed_N)

        assert res.dtype == transform.transformed_dtype()
        assert decaFFT.dtype == transform.transformed_dtype()
        assert gsw.dtype == transform.transformed_dtype()

        d = decaFFT.reshape(batch_shape + (k+1, l, 1, transformed_N))
        res.fill(0)
        for i in range(k + 1):
            for j in range(l):
                res[:,:,:] = transform.transformed_space_add_ref(
                    res, transform.transformed_space_mul_ref(d[:,i,j,:,:], gsw[bk_idx,i,j,:,:]))

    return _kernel


# External product (*): accum = gsw (*) accum
def TGswFFTExternMulToTLwe_reference(accum_a, gsw, params: 'TGswParams'):

    transform = get_transform(params.tlwe_params.transform_type)

    def _kernel(accum_a, gsw, bk_idx):

        tlwe_params = params.tlwe_params
        k = tlwe_params.mask_size
        l = params.decomp_length
        N = tlwe_params.polynomial_degree

        batch_shape = accum_a.shape[:-2]
        deca = numpy.empty(batch_shape + (k + 1, l, N), numpy.int32)
        tmpa_a = numpy.empty(
            batch_shape + (k + 1, transform.transformed_length(N)), transform.transformed_dtype())

        TGswTorus32PolynomialDecompH_reference(deca, params)(deca, accum_a)

        decaFFT = transform.forward_transform_ref(deca)

        TLweFFTAddMulRTo_reference(tmpa_a, gsw, tlwe_params)(tmpa_a, decaFFT, gsw, bk_idx)

        numpy.copyto(accum_a, transform.inverse_transform_ref(tmpa_a))

    return _kernel


# Result += mu*H, mu integer
def TGswAddMuIntH_ref(n, params: 'TGswParams'):
    # TYPING: messages::Array{Int32, 1}
    k = params.tlwe_params.mask_size
    l = params.decomp_length
    h = params.base_powers

    def _kernel(result_a, messages):

        # compute result += H

        # returns an underlying coefsT of TorusPolynomialArray, with the total size
        # (N, k + 1 [from TLweSample], l, k + 1 [from TGswSample], n)
        # messages: (n,)
        # h: (l,)
        # TODO: use an appropriate method
        # TODO: not sure if it's possible to fully vectorize it
        for bloc in range(k+1):
            result_a[:, bloc, :, bloc, 0] += (
                messages.reshape(messages.size, 1) * h.reshape(1, l))

    return _kernel
