import numpy

from reikna.algorithms import PureParallel

from tfhe.tgsw import TGswParams, TGswSampleArray, TGswSampleFFTArray
from tfhe.tlwe import TLweSampleArray
from tfhe.keys import TFHEParameters
from tfhe.numeric_functions import Torus32
from tfhe.polynomial_transform import (
    forward_transform_ref, inverse_transform_ref, transformed_internal_ctype,
    transformed_dtype, transformed_length, transformed_space_mul_ref, transformed_space_add_ref)
from tfhe.gpu_tgsw import (
    get_TGswTorus32PolynomialDecompH_trf, get_TLweFFTAddMulRTo_trf, TGswFFTExternMulToTLwe,
    TGswAddMuIntH, TGswAddMuIntH_ref)


def TGswTorus32PolynomialDecompH_reference(_result, params: TGswParams):

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


def test_TGswTorus32PolynomialDecompH(thread):

    batch = (2, 3)
    params = TFHEParameters()
    tgsw_params = params.tgsw_params
    l = tgsw_params.l
    k = tgsw_params.tlwe_params.k
    N = tgsw_params.tlwe_params.N

    sample = numpy.random.randint(0, 1000, size=batch + (k + 1, N), dtype=Torus32)
    result = numpy.empty(batch + (k + 1, l, N), dtype=numpy.int32)

    sample_dev = thread.to_device(sample)
    result_dev = thread.empty_like(result)

    trf = get_TGswTorus32PolynomialDecompH_trf(result, tgsw_params)
    test = PureParallel.from_trf(trf, guiding_array='output').compile(thread)

    ref = TGswTorus32PolynomialDecompH_reference(result, tgsw_params)

    test(result_dev, sample_dev)
    result_test = result_dev.get()

    ref(result, sample)

    assert (result == result_test).all()



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


def test_TLweFFTAddMulRTo(thread):

    batch = (2,)
    params = TFHEParameters()
    tgsw_params = params.tgsw_params

    l = tgsw_params.l
    k = tgsw_params.tlwe_params.k
    N = tgsw_params.tlwe_params.N

    tmpa_a_shape = batch + (k + 1, transformed_length(N))
    decaFFT_shape = batch + (k + 1, l, transformed_length(N))
    gsw_shape = (10, k + 1, l, k + 1, transformed_length(N))
    bk_idx = 2

    tmpa_a = numpy.empty(tmpa_a_shape, transformed_dtype())

    if transformed_dtype().kind == 'c':
        decaFFT = (
            numpy.random.normal(size=decaFFT_shape)
            + 1j * numpy.random.normal(size=decaFFT_shape)).astype(transformed_dtype())
        gsw = (
            numpy.random.normal(size=gsw_shape)
            + 1j * numpy.random.normal(size=gsw_shape)).astype(transformed_dtype())
    else:
        decaFFT = numpy.random.randint(0, 2**64-2**32+1, size=decaFFT_shape, dtype=transformed_dtype())
        gsw = numpy.random.randint(0, 2**64-2**32+1, size=gsw_shape, dtype=transformed_dtype())

    tmpa_a_dev = thread.empty_like(tmpa_a)
    decaFFT_dev = thread.to_device(decaFFT)
    gsw_dev = thread.to_device(gsw)

    trf = get_TLweFFTAddMulRTo_trf(N, tmpa_a, gsw, transformed_internal_ctype())
    test = PureParallel.from_trf(trf, guiding_array='tmpa_a').compile(thread)
    ref = TLweFFTAddMulRTo_reference(tmpa_a, gsw)

    test(tmpa_a_dev, decaFFT_dev, gsw_dev, bk_idx)
    tmpa_a_test = tmpa_a_dev.get()

    ref(tmpa_a, decaFFT, gsw, bk_idx)

    assert numpy.allclose(tmpa_a, tmpa_a_test)


# External product (*): accum = gsw (*) accum
def TGswFFTExternMulToTLwe_reference(accum_a, gsw, params: TGswParams):

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


def test_TGswFFTExternMulToTLwe(thread):

    batch = (16,)
    params = TFHEParameters()
    tgsw_params = params.tgsw_params

    l = tgsw_params.l
    k = tgsw_params.tlwe_params.k
    N = tgsw_params.tlwe_params.N

    accum_a_shape = batch + (k + 1, N)
    gsw_shape = (10, k + 1, l, k + 1, transformed_length(N))
    bk_idx = 2

    if transformed_dtype().kind == 'c':
        gsw = (numpy.random.normal(size=gsw_shape)
            + 1j * numpy.random.normal(size=gsw_shape)).astype(transformed_dtype()) * 1000
    else:
        gsw = numpy.random.randint(0, 1000, size=gsw_shape, dtype=transformed_dtype())

    accum_a = numpy.random.randint(-1000, 1000, size=accum_a_shape, dtype=Torus32)

    gsw_dev = thread.to_device(gsw)
    accum_a_dev = thread.to_device(accum_a)

    test = TGswFFTExternMulToTLwe(accum_a, gsw, tgsw_params).compile(thread)
    ref = TGswFFTExternMulToTLwe_reference(accum_a, gsw, tgsw_params)

    test(accum_a_dev, gsw_dev, bk_idx)
    accum_a_test = accum_a_dev.get()

    ref(accum_a, gsw, bk_idx)

    assert numpy.allclose(accum_a, accum_a_test)


def test_TGswAddMuIntH(thread):

    params = TFHEParameters()
    tgsw_params = params.tgsw_params
    n = params.in_out_params.n
    l = tgsw_params.l
    k = tgsw_params.tlwe_params.k
    N = tgsw_params.tlwe_params.N

    result_a = numpy.random.randint(-2**31, 2**31, size=(n, k+1, l, k+1, N), dtype=Torus32)
    messages = numpy.random.randint(-2**31, 2**31, size=(n,), dtype=Torus32)

    result_a_dev = thread.to_device(result_a)
    messages_dev = thread.to_device(messages)

    test = TGswAddMuIntH(n, tgsw_params).compile(thread)
    ref = TGswAddMuIntH_ref(n, tgsw_params)

    test(result_a_dev, messages_dev)
    ref(result_a, messages)

    result_a_test = result_a_dev.get()
    messages_test = messages_dev.get()

    assert numpy.allclose(result_a_test, result_a)
