import numpy

from reikna.cluda import functions, Module

from .transform import fft512, Transform
from .transform.fft import fft_transform_ref
from .performance import PerformanceParameters


def transformed_dtype():
    return numpy.dtype('complex128')


def transformed_internal_dtype():
    return numpy.dtype('complex128')


elem =  Module.create(
    lambda prefix: """
    typedef double2 ${prefix};

    #define ${prefix}pack(x) (x)
    #define ${prefix}unpack(x) (x)
    #define ${prefix}zero (COMPLEX_CTR(double2)(0, 0))
    """,
    render_kwds=dict())


def transformed_internal_ctype():
    return elem


def transformed_length(N):
    return N // 2


def forward_transform_ref(data):
    return fft_transform_ref(data, i32_conversion=True)


def inverse_transform_ref(data):
    return fft_transform_ref(data, i32_conversion=True, inverse=True)


def transformed_space_add_ref(data1, data2):
    return data1 + data2


def transformed_space_mul_ref(data1, data2):
    return data1 * data2


def transformed_add(perf_params):
    return functions.add(transformed_dtype(), transformed_dtype())


def transformed_mul(perf_params):
    return functions.mul(transformed_dtype(), transformed_dtype())


def transform_module(perf_params: PerformanceParameters, multi_iter=False):
    use_constant_memory = (
        perf_params.use_constant_memory_multi_iter if multi_iter
        else perf_params.use_constant_memory_single_iter)
    return fft512(use_constant_memory=use_constant_memory)


def ForwardTransform(batch_shape, N, perf_params: PerformanceParameters):
    assert N == 1024
    return Transform(
        transform_module(perf_params), batch_shape,
        transforms_per_block=perf_params.transforms_per_block, i32_conversion=True)


def InverseTransform(batch_shape, N, perf_params: PerformanceParameters):
    assert N == 1024
    return Transform(
        transform_module(perf_params), batch_shape,
        transforms_per_block=perf_params.transforms_per_block, i32_conversion=True, inverse=True)
