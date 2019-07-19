# Copyright (C) 2018 NuCypher
#
# This file is part of nufhe.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy

from reikna.core import Transformation, Parameter, Annotation, Type

from .transform import ntt1024, ntt1024_requirements, Transform
from .transform.arithmetic import add, mul, get_ff_elem, prepare_for_mul, mul_prepared
from .transform.ntt import ntt_transform_ref
from .transform import ntt_cpu
from .performance import PerformanceParametersForDevice


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


def transformed_space_mul_prepared_ref(data1, data2):
    coeff = ntt_cpu.gnum(0xfffffffe00000001) # Inverse of 2**64 modulo (2**64-2**32+1)
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 * data2 * coeff)


ff_elem = get_ff_elem()


def transformed_add(perf_params: PerformanceParametersForDevice):
    return add(ff_elem=ff_elem, method=perf_params.ntt_base_method).module


def transformed_mul(perf_params: PerformanceParametersForDevice):
    return mul(ff_elem=ff_elem, method=perf_params.ntt_mul_method).module


def transformed_mul_prepared(perf_params: PerformanceParametersForDevice):
    return mul_prepared(ff_elem=ff_elem, method=perf_params.ntt_mul_method).module


def transform_module_requirements():
    return ntt1024_requirements()


def get_prepare_for_mul_trf(shape):
    dtype = transformed_dtype()
    return Transformation([
        Parameter('output', Annotation(Type(dtype, shape), 'o')),
        Parameter('input', Annotation(Type(dtype, shape), 'i'))],
        """
        ${dtypes.ctype(dtype)} x = ${input.load_same};
        ${ff_ctype} x_ff = { x };
        ${output.store_same}(${prepare_for_mul}(x_ff).val);
        """,
        connectors=['input', 'output'],
        render_kwds=dict(
            prepare_for_mul=prepare_for_mul(ff_elem=ff_elem).module,
            dtype=dtype,
            ff_ctype=transformed_internal_ctype()))


def transform_module(perf_params: PerformanceParametersForDevice, multi_iter=False):
    use_constant_memory = (
        perf_params.use_constant_memory_multi_iter if multi_iter
        else perf_params.use_constant_memory_single_iter)
    return ntt1024(
        ff_elem=ff_elem,
        base_method=perf_params.ntt_base_method,
        mul_method=perf_params.ntt_mul_method,
        lsh_method=perf_params.ntt_lsh_method,
        use_constant_memory=use_constant_memory)


def ForwardTransform(batch_shape, N, perf_params: PerformanceParametersForDevice):
    assert N == 1024
    return Transform(
        transform_module(perf_params), batch_shape,
        transforms_per_block=perf_params.transforms_per_block, i32_conversion=True)


def InverseTransform(batch_shape, N, perf_params: PerformanceParametersForDevice):
    assert N == 1024
    return Transform(
        transform_module(perf_params), batch_shape,
        transforms_per_block=perf_params.transforms_per_block, i32_conversion=True, inverse=True)
