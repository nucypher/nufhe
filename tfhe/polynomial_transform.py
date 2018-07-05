import numpy

from reikna.cluda import functions

from .transform import ntt1024, Transform
from .transform.arithmetic import add, mul, get_ff_elem
from .transform.ntt import ntt_transform_ref
from .transform import ntt_cpu


def transformed_dtype():
    return numpy.dtype('uint64')


def transformed_internal_dtype():
    return numpy.dtype([("val", numpy.uint64)])


def transformed_internal_ctype():
    return ff_elem.module


def transformed_length(N):
    return N


def forward_transform_ref(data):
    return ntt_transform_ref(data, i32_conversion=True)


def inverse_transform_ref(data):
    return ntt_transform_ref(data, i32_conversion=True, inverse=True)


def transformed_space_add_ref(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 + data2)


def transformed_space_mul_ref(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 * data2)


ff_elem = get_ff_elem()


def transformed_add():
    return add(ff_elem=ff_elem, method='cuda_asm').module


def transformed_mul():
    return mul(ff_elem=ff_elem, method='cuda_asm').module


def transform_module():
    return ntt1024(
        ff_elem=ff_elem,
        base_method='cuda_asm', mul_method='cuda_asm', lsh_method='cuda_asm',
        use_constant_memory=True)


def ForwardTransform(batch_shape, N):
    assert N == 1024
    return Transform(
        transform_module(), batch_shape, transforms_per_block=1, i32_conversion=True)


def InverseTransform(batch_shape, N):
    assert N == 1024
    return Transform(
        transform_module(), batch_shape, transforms_per_block=1, i32_conversion=True, inverse=True)
