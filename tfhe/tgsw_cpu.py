import numpy

from tfhe.numeric_functions import Torus32
from tfhe.polynomial_transform import (
    forward_transform_ref, inverse_transform_ref,
    transformed_dtype, transformed_length, transformed_space_mul_ref, transformed_space_add_ref)


def TGswTorus32PolynomialDecompH_reference(_result, params: 'TGswParams'):

    def _kernel(result, sample):

        N = result.shape[-1]
        l = result.shape[-2]
        k = result.shape[-3] - 1
        batch = result.shape[:-3]

        assert sample.shape == batch + (k + 1, N)
        assert result.dtype == numpy.int32
        assert sample.dtype == Torus32

        Bgbit = params.Bgbit
        maskMod = params.maskMod
        halfBg = params.halfBg
        offset = params.offset

        decal = lambda p: 32 - p * Bgbit

        ps = numpy.arange(1, l+1).reshape((1,) * len(batch) + (1, l, 1))
        sample_coefs = sample.reshape(sample.shape[:-1] + (1, N))

        # do the decomposition
        numpy.copyto(result, (((sample_coefs + offset) >> decal(ps)) & maskMod) - halfBg)

    return _kernel


def TLweFFTAddMulRTo_reference(res, gsw):

    def _kernel(res, decaFFT, gsw, bk_idx):

        batch_shape = res.shape[:-2]
        k = res.shape[-2] - 1
        transformed_N = res.shape[-1]
        l = decaFFT.shape[-2]

        assert decaFFT.shape == batch_shape + (k + 1, l, transformed_N)
        assert gsw.shape[-4:] == (k + 1, l, k + 1, transformed_N)
        assert res.shape == batch_shape + (k + 1, transformed_N)

        assert res.dtype == transformed_dtype()
        assert decaFFT.dtype == transformed_dtype()
        assert gsw.dtype == transformed_dtype()

        d = decaFFT.reshape(batch_shape + (k+1, l, 1, transformed_N))
        res.fill(0)
        for i in range(k + 1):
            for j in range(l):
                res[:,:,:] = transformed_space_add_ref(
                    res, transformed_space_mul_ref(d[:,i,j,:,:], gsw[bk_idx,i,j,:,:]))

    return _kernel


# External product (*): accum = gsw (*) accum
def TGswFFTExternMulToTLwe_reference(accum_a, gsw, params: 'TGswParams'):

    def _kernel(accum_a, gsw, bk_idx):

        tlwe_params = params.tlwe_params
        k = tlwe_params.k
        l = params.l
        kpl = params.kpl
        N = tlwe_params.N

        batch_shape = accum_a.shape[:-2]
        deca = numpy.empty(batch_shape + (k + 1, l, N), numpy.int32)
        tmpa_a = numpy.empty(batch_shape + (k + 1, transformed_length(N)), transformed_dtype())

        TGswTorus32PolynomialDecompH_reference(deca, params)(deca, accum_a)

        decaFFT = forward_transform_ref(deca)

        TLweFFTAddMulRTo_reference(tmpa_a, gsw)(tmpa_a, decaFFT, gsw, bk_idx)

        numpy.copyto(accum_a, inverse_transform_ref(tmpa_a))

    return _kernel


# Result += mu*H, mu integer
def TGswAddMuIntH_ref(n, params: 'TGswParams'):
    # TYPING: messages::Array{Int32, 1}
    k = params.tlwe_params.k
    l = params.l
    h = params.h

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
