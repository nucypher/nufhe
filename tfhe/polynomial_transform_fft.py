import numpy

from reikna.cluda import functions, Module

from .transform import fft512, Transform
from .transform.fft import fft_transform_ref


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


def transformed_add():
    return functions.add(transformed_dtype(), transformed_dtype())


def transformed_mul():
    return functions.mul(transformed_dtype(), transformed_dtype())


def transform_module():
    return fft512(use_constant_memory=True)


def ForwardTransform(batch_shape, N):
    assert N == 1024
    return Transform(
        fft512(), batch_shape, transforms_per_block=1, i32_conversion=True)


def InverseTransform(batch_shape, N):
    assert N == 1024
    return Transform(
        fft512(), batch_shape, transforms_per_block=1, i32_conversion=True, inverse=True)
