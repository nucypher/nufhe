import numpy

from .transform import ntt1024, Transform
from .transform.arithmetic import add, mul, get_ff_elem
from .transform.ntt import ntt_transform_ref
from .transform import ntt_cpu
from .performance import PerformanceParameters


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


def transformed_add(perf_params: PerformanceParameters):
    return add(ff_elem=ff_elem, method=perf_params.ntt_base_method).module


def transformed_mul(perf_params: PerformanceParameters):
    return mul(ff_elem=ff_elem, method=perf_params.ntt_mul_method).module


def transform_module(perf_params: PerformanceParameters):
    return ntt1024(
        ff_elem=ff_elem,
        base_method=perf_params.ntt_base_method,
        mul_method=perf_params.ntt_mul_method,
        lsh_method=perf_params.ntt_lsh_method,
        use_constant_memory=perf_params.use_constant_memory)


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
