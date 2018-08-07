import numpy

from reikna.algorithms import PureParallel

from tfhe.tgsw import TGswParams, TGswSampleArray, TransformedTGswSampleArray
from tfhe.tlwe import TLweSampleArray
from tfhe.keys import TFHEParameters
from tfhe.numeric_functions import Torus32, Int32
from tfhe.polynomial_transform import get_transform
from tfhe.tgsw_gpu import (
    get_tgsw_polynomial_decomp_trf,
    get_tlwe_transformed_add_mul_to_trf,
    TGswTransformedExternalMul,
    TGswAddMessage,
    )
from tfhe.tgsw_cpu import (
    tgsw_polynomial_decomp_trf_reference,
    tlwe_transformed_add_mul_to_trf_reference,
    TGswTransformedExternalMulReference,
    TGswAddMessageReference,
    )
from tfhe.performance import performance_parameters

from utils import get_test_array


def test_tgsw_polynomial_decomp_trf(thread):

    shape = (2, 3)
    params = TFHEParameters()
    tgsw_params = params.tgsw_params
    decomp_length = tgsw_params.decomp_length
    mask_size = tgsw_params.tlwe_params.mask_size
    polynomial_degree = tgsw_params.tlwe_params.polynomial_degree

    sample = get_test_array(shape + (mask_size + 1, polynomial_degree), Torus32, (0, 1000))
    result = numpy.empty(shape + (mask_size + 1, decomp_length, polynomial_degree), dtype=Int32)

    sample_dev = thread.to_device(sample)
    result_dev = thread.empty_like(result)

    trf = get_tgsw_polynomial_decomp_trf(tgsw_params, shape)
    test = PureParallel.from_trf(trf, guiding_array='result').compile(thread)

    ref = tgsw_polynomial_decomp_trf_reference(tgsw_params, shape)

    test(result_dev, sample_dev)
    result_test = result_dev.get()

    ref(result, sample)

    assert (result == result_test).all()


def test_tlwe_transformed_add_mul_to_trf(thread):

    shape = (2, 3)
    params = TFHEParameters()
    perf_params = performance_parameters()
    tgsw_params = params.tgsw_params

    decomp_length = tgsw_params.decomp_length
    mask_size = tgsw_params.tlwe_params.mask_size
    polynomial_degree = tgsw_params.tlwe_params.polynomial_degree

    transform_type = tgsw_params.tlwe_params.transform_type
    transform = get_transform(transform_type)
    tlength = transform.transformed_length(polynomial_degree)
    tdtype = transform.transformed_dtype()

    result_shape = shape + (mask_size + 1, tlength)
    sample_shape = shape + (mask_size + 1, decomp_length, tlength)
    bk_len = 10
    bootstrap_key_shape = (bk_len, mask_size + 1, decomp_length, mask_size + 1, tlength)
    bk_row_idx = 2

    result = numpy.empty(result_shape, tdtype)

    sample = get_test_array(sample_shape, 'ff_number' if transform_type == 'NTT' else tdtype)
    bootstrap_key = get_test_array(
        bootstrap_key_shape, 'ff_number' if transform_type == 'NTT' else tdtype)

    result_dev = thread.empty_like(result)
    sample_dev = thread.to_device(sample)
    bootstrap_key_dev = thread.to_device(bootstrap_key)

    trf = get_tlwe_transformed_add_mul_to_trf(tgsw_params, shape, bk_len, perf_params)
    test = PureParallel.from_trf(trf, guiding_array='result').compile(thread)
    ref = tlwe_transformed_add_mul_to_trf_reference(tgsw_params, shape, bk_len, perf_params)

    test(result_dev, sample_dev, bootstrap_key_dev, bk_row_idx)
    result_test = result_dev.get()

    ref(result, sample, bootstrap_key, bk_row_idx)

    assert numpy.allclose(result, result_test)


def test_tgsw_transformed_external_mul(thread):

    shape = (2, 3)
    params = TFHEParameters()
    perf_params = performance_parameters()
    tgsw_params = params.tgsw_params

    decomp_length = tgsw_params.decomp_length
    mask_size = tgsw_params.tlwe_params.mask_size
    polynomial_degree = tgsw_params.tlwe_params.polynomial_degree

    transform_type = tgsw_params.tlwe_params.transform_type
    transform = get_transform(transform_type)
    tlength = transform.transformed_length(polynomial_degree)
    tdtype = transform.transformed_dtype()

    accum_shape = shape + (mask_size + 1, polynomial_degree)
    bk_len = 10
    bootstrap_key_shape = (bk_len, mask_size + 1, decomp_length, mask_size + 1, tlength)
    bk_row_idx = 2

    bootstrap_key = get_test_array(
        bootstrap_key_shape, 'ff_number' if transform_type == 'NTT' else tdtype)
    accum = get_test_array(accum_shape, Torus32, (-1000, 1000))

    bootstrap_key_dev = thread.to_device(bootstrap_key)
    accum_dev = thread.to_device(accum)

    test = TGswTransformedExternalMul(tgsw_params, shape, bk_len, perf_params).compile(thread)
    ref = TGswTransformedExternalMulReference(tgsw_params, shape, bk_len, perf_params)

    test(accum_dev, bootstrap_key_dev, bk_row_idx)
    accum_test = accum_dev.get()

    ref(accum, bootstrap_key, bk_row_idx)

    assert numpy.allclose(accum, accum_test)


def test_tgsw_add_message(thread):

    params = TFHEParameters()
    tgsw_params = params.tgsw_params

    decomp_length = tgsw_params.decomp_length
    mask_size = tgsw_params.tlwe_params.mask_size
    polynomial_degree = tgsw_params.tlwe_params.polynomial_degree

    shape = (3, 5)

    result_a = get_test_array(
        shape + (mask_size + 1, decomp_length, mask_size + 1, polynomial_degree), Torus32)
    messages = get_test_array(shape, Torus32)

    result_a_dev = thread.to_device(result_a)
    messages_dev = thread.to_device(messages)

    test = TGswAddMessage(tgsw_params, shape).compile(thread)
    ref = TGswAddMessageReference(tgsw_params, shape)

    test(result_a_dev, messages_dev)
    ref(result_a, messages)

    result_a_test = result_a_dev.get()

    assert numpy.allclose(result_a_test, result_a)
