import numpy

from reikna.algorithms import PureParallel

from tfhe.tgsw import TGswParams, TGswSampleArray, TGswSampleFFTArray
from tfhe.tlwe import TLweSampleArray
from tfhe.keys import TFHEParameters
from tfhe.numeric_functions import Torus32
from tfhe.polynomial_transform import get_transform
from tfhe.tgsw_gpu import (
    get_TGswTorus32PolynomialDecompH_trf,
    get_TLweFFTAddMulRTo_trf,
    TGswFFTExternMulToTLwe,
    TGswAddMuIntH,
    )
from tfhe.tgsw_cpu import (
    TGswAddMuIntH_ref,
    TGswTorus32PolynomialDecompH_reference,
    TLweFFTAddMulRTo_reference,
    TGswFFTExternMulToTLwe_reference,
    )
from tfhe.performance import performance_parameters


def test_TGswTorus32PolynomialDecompH(thread):

    batch = (2, 3)
    params = TFHEParameters()
    tgsw_params = params.tgsw_params
    l = tgsw_params.decomp_length
    k = tgsw_params.tlwe_params.mask_size
    N = tgsw_params.tlwe_params.polynomial_degree

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


def test_TLweFFTAddMulRTo(thread):

    batch = (2,)
    params = TFHEParameters()
    perf_params = performance_parameters()
    tgsw_params = params.tgsw_params

    transform_type = tgsw_params.tlwe_params.transform_type
    transform = get_transform(transform_type)

    l = tgsw_params.decomp_length
    k = tgsw_params.tlwe_params.mask_size
    N = tgsw_params.tlwe_params.polynomial_degree

    tr_N = transform.transformed_length(N)
    tr_dtype = transform.transformed_dtype()

    tmpa_a_shape = batch + (k + 1, tr_N)
    decaFFT_shape = batch + (k + 1, l, tr_N)
    gsw_shape = (10, k + 1, l, k + 1, tr_N)
    bk_idx = 2

    tmpa_a = numpy.empty(tmpa_a_shape, tr_dtype)

    if tr_dtype.kind == 'c':
        decaFFT = (
            numpy.random.normal(size=decaFFT_shape)
            + 1j * numpy.random.normal(size=decaFFT_shape)).astype(tr_dtype)
        gsw = (
            numpy.random.normal(size=gsw_shape)
            + 1j * numpy.random.normal(size=gsw_shape)).astype(tr_dtype)
    else:
        decaFFT = numpy.random.randint(0, 2**64-2**32+1, size=decaFFT_shape, dtype=tr_dtype)
        gsw = numpy.random.randint(0, 2**64-2**32+1, size=gsw_shape, dtype=tr_dtype)

    tmpa_a_dev = thread.empty_like(tmpa_a)
    decaFFT_dev = thread.to_device(decaFFT)
    gsw_dev = thread.to_device(gsw)

    trf = get_TLweFFTAddMulRTo_trf(
        N, transform_type, tmpa_a, gsw, transform.transformed_internal_ctype(),
        perf_params)
    test = PureParallel.from_trf(trf, guiding_array='tmpa_a').compile(thread)
    ref = TLweFFTAddMulRTo_reference(tmpa_a, gsw, tgsw_params.tlwe_params)

    test(tmpa_a_dev, decaFFT_dev, gsw_dev, bk_idx)
    tmpa_a_test = tmpa_a_dev.get()

    ref(tmpa_a, decaFFT, gsw, bk_idx)

    assert numpy.allclose(tmpa_a, tmpa_a_test)


def test_TGswFFTExternMulToTLwe(thread):

    batch = (16,)
    params = TFHEParameters()
    perf_params = performance_parameters()
    tgsw_params = params.tgsw_params

    transform_type = tgsw_params.tlwe_params.transform_type
    transform = get_transform(transform_type)

    l = tgsw_params.decomp_length
    k = tgsw_params.tlwe_params.mask_size
    N = tgsw_params.tlwe_params.polynomial_degree

    accum_a_shape = batch + (k + 1, N)
    gsw_shape = (10, k + 1, l, k + 1, transform.transformed_length(N))
    bk_idx = 2

    tr_dtype = transform.transformed_dtype()

    if tr_dtype.kind == 'c':
        gsw = (numpy.random.normal(size=gsw_shape)
            + 1j * numpy.random.normal(size=gsw_shape)).astype(tr_dtype) * 1000
    else:
        gsw = numpy.random.randint(0, 1000, size=gsw_shape, dtype=tr_dtype)

    accum_a = numpy.random.randint(-1000, 1000, size=accum_a_shape, dtype=Torus32)

    gsw_dev = thread.to_device(gsw)
    accum_a_dev = thread.to_device(accum_a)

    test = TGswFFTExternMulToTLwe(accum_a, gsw, tgsw_params, perf_params).compile(thread)
    ref = TGswFFTExternMulToTLwe_reference(accum_a, gsw, tgsw_params, perf_params)

    test(accum_a_dev, gsw_dev, bk_idx)
    accum_a_test = accum_a_dev.get()

    ref(accum_a, gsw, bk_idx)

    assert numpy.allclose(accum_a, accum_a_test)


def test_TGswAddMuIntH(thread):

    params = TFHEParameters()
    tgsw_params = params.tgsw_params
    n = params.in_out_params.size
    l = tgsw_params.decomp_length
    k = tgsw_params.tlwe_params.mask_size
    N = tgsw_params.tlwe_params.polynomial_degree

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
