import numpy

from tfhe.tgsw import TGswParams, TGswSampleArray, TGswSampleFFTArray
from tfhe.tlwe import TLweSampleArray
from tfhe.keys import TFHEParameters
from tfhe.numeric_functions import Torus32, Complex

from tfhe.gpu_tgsw import TGswTorus32PolynomialDecompH, TLweFFTAddMulRTo, TGswFFTExternMulToTLwe

from test_polynomials import ip_ifft_reference, tp_fft_reference


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

    test = TGswTorus32PolynomialDecompH(result, tgsw_params).compile(thread)
    ref = TGswTorus32PolynomialDecompH_reference(result, tgsw_params)

    test(result_dev, sample_dev)
    result_test = result_dev.get()

    ref(result, sample)

    assert (result == result_test).all()



def TLweFFTAddMulRTo_reference(res, gsw):

    def _kernel(res, decaFFT, gsw, bk_idx):

        batch_shape = res.shape[:-2]
        k = res.shape[-2] - 1
        N = res.shape[-1] * 2
        l = decaFFT.shape[-2]

        assert decaFFT.shape == batch_shape + (k + 1, l, N // 2)
        assert gsw.shape[-4:] == (k + 1, l, k + 1, N // 2)
        assert res.shape == batch_shape + (k + 1, N // 2)

        assert res.dtype == Complex
        assert decaFFT.dtype == Complex
        assert gsw.dtype == Complex

        d = decaFFT.reshape(batch_shape + (k+1, l, 1, N//2))
        res.fill(0)
        for i in range(k + 1):
            for j in range(l):
                res += d[:,i,j,:,:] * gsw[bk_idx,i,j,:,:]

    return _kernel


def test_TLweFFTAddMulRTo(thread):

    batch = (2,)
    params = TFHEParameters()
    tgsw_params = params.tgsw_params

    l = tgsw_params.l
    k = tgsw_params.tlwe_params.k
    N = tgsw_params.tlwe_params.N

    tmpa_a_shape = batch + (k + 1, N//2)
    decaFFT_shape = batch + (k + 1, l, N//2)
    gsw_shape = (10, k + 1, l, k + 1, N//2)
    bk_idx = 2

    tmpa_a = numpy.empty(tmpa_a_shape, Complex)
    decaFFT = numpy.random.normal(size=decaFFT_shape) + 1j * numpy.random.normal(size=decaFFT_shape)
    gsw = numpy.random.normal(size=gsw_shape) + 1j * numpy.random.normal(size=gsw_shape)

    tmpa_a_dev = thread.empty_like(tmpa_a)
    decaFFT_dev = thread.to_device(decaFFT)
    gsw_dev = thread.to_device(gsw)

    test = TLweFFTAddMulRTo(tmpa_a, gsw).compile(thread)
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
        decaFFT = numpy.empty(batch_shape + (k + 1, l, N // 2), Complex)
        tmpa_a = numpy.empty(batch_shape + (k + 1, N // 2), Complex)

        TGswTorus32PolynomialDecompH_reference(deca, params)(deca, accum_a)

        ip_ifft_reference(decaFFT, deca, 2)

        TLweFFTAddMulRTo_reference(tmpa_a, gsw)(tmpa_a, decaFFT, gsw, bk_idx)

        tp_fft_reference(accum_a, tmpa_a)

    return _kernel


def test_TGswFFTExternMulToTLwe(thread):

    batch = (16,)
    params = TFHEParameters()
    tgsw_params = params.tgsw_params

    l = tgsw_params.l
    k = tgsw_params.tlwe_params.k
    N = tgsw_params.tlwe_params.N

    accum_a_shape = batch + (k + 1, N)
    gsw_shape = (10, k + 1, l, k + 1, N//2)
    bk_idx = 2

    gsw = numpy.random.normal(size=gsw_shape) + 1j * numpy.random.normal(size=gsw_shape)
    accum_a = numpy.random.randint(-1000, 1000, size=accum_a_shape, dtype=Torus32)

    gsw_dev = thread.to_device(gsw)
    accum_a_dev = thread.to_device(accum_a)

    test = TGswFFTExternMulToTLwe(accum_a, gsw, tgsw_params).compile(thread)
    ref = TGswFFTExternMulToTLwe_reference(accum_a, gsw, tgsw_params)

    test(accum_a_dev, gsw_dev, bk_idx)
    accum_a_test = accum_a_dev.get()

    ref(accum_a, gsw, bk_idx)

    assert numpy.allclose(accum_a, accum_a_test)
