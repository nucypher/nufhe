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

import time
import numpy
import pytest

import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.helpers import product

import nufhe.transform as ntt
import nufhe.transform.ntt_cpu as ntt_cpu

from utils import get_test_array, tp_dtype


@pytest.fixture(params=["c", "cuda_asm", "c_from_asm"])
def method(request):
    return request.param


def get_func_kernel(thread, func_module, output_type, input_types, repetitions=1):
    src = """
    <%
        argnames = ["a" + str(i + 1) for i in range(len(input_types))]
        ctype = lambda x: str(func.ff.module) if x == 'ff_number' else dtypes.ctype(x)
    %>
    KERNEL void test(
        GLOBAL_MEM ${ctype(output_type)} *dest
        %for arg, tp in zip(argnames, input_types):
        , GLOBAL_MEM ${ctype(tp)} *${arg}
        %endfor
        )
    {
        const SIZE_T i = get_global_id(0);
        %for arg, tp in zip(argnames, input_types):
        ${ctype(tp)} ${arg}_load = ${arg}[i];
        %endfor

        // To stop the compiler from optimizing away the code,
        // we need to use the return value somehow.
        // We're using it as the first argument to the next invocation.
        // Assuming here that the first argument of the tested function is either
        // a finite field element, or something convertable to it (e.g. uint64).
        ${ctype(output_type)} res =
        %if ctype(input_types[0]) == ctype(output_type):
            ${argnames[0]}_load;
        %else:
            { ${argnames[0]}_load };
        %endif

        for (int i = 0; i < ${repetitions}; i++)
        {
            res = ${func.module}(
                %if ctype(input_types[0]) == ctype(output_type):
                res
                %else:
                res.val
                %endif
                %if len(argnames) > 1:
                , ${", ".join([arg + "_load" for arg in argnames[1:]])}
                %endif
                );
        }

        dest[i] = res;
    }
    """

    program = thread.compile(
        src,
        render_kwds=dict(
            input_types=input_types, output_type=output_type, func=func_module,
            repetitions=repetitions))

    return program.test


def check_func(
        thread, func_module, reference_func, output_type, input_types,
        ranges=None, test_values=None):

    N = 1024

    test = get_func_kernel(thread, func_module, output_type, input_types)

    arrays = [
        get_test_array(N, tp, val_range=ranges[i] if ranges is not None else None)
        for i, tp in enumerate(input_types)]

    if test_values is not None:
        for i, tvs in enumerate(test_values):
            if tvs is not None:
                for j, tv in enumerate(tvs):
                    arrays[j][i] = tv

    arrays_dev = [thread.to_device(arr) for arr in arrays]
    dest_dev = thread.array(N, tp_dtype(output_type))

    test(dest_dev, *arrays_dev, global_size=N)

    assert (dest_dev.get() == reference_func(*arrays)).all()


def ref_add(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 + data2)


def test_add(thread, method):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    check_func(thread, ntt.add(method=method), ref_add, 'ff_number', ['ff_number', 'ff_number'])


def ref_sub(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 - data2)


def test_sub(thread, method):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    check_func(thread, ntt.sub(method=method), ref_sub, 'ff_number', ['ff_number', 'ff_number'])


def ref_mod(data):
    return data % ntt_cpu.GaloisNumber.modulus


def test_mod(thread, method):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    check_func(
        thread, ntt.mod(method=method), ref_mod, 'ff_number', [numpy.uint64],
        test_values=[
            (ntt_cpu.GaloisNumber.modulus - 1,),
            (ntt_cpu.GaloisNumber.modulus,),
            (ntt_cpu.GaloisNumber.modulus + 1,)])


def ref_mul(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 * data2)


def test_mul(thread, method):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    check_func(
        thread, ntt.mul(method=method), ref_mul, 'ff_number', ['ff_number', 'ff_number'],
        test_values=[
            (ntt_cpu.GaloisNumber.modulus - 1, 2**33) # regression test for an error in method=c
            ]
        )


def ref_mul_prepared(data1, data2):
    coeff = ntt_cpu.gnum(0xfffffffe00000001) # Inverse of 2**64 modulo (2**64-2**32+1)
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 * data2 * coeff)


def test_mul_prepared(thread, method):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    check_func(
        thread, ntt.mul_prepared(method=method), ref_mul_prepared,
        'ff_number', ['ff_number', 'ff_number'])


def ref_prepare_for_mul(data1):
    coeff = ntt_cpu.gnum(0xffffffff) # 2**64 modulo (2**64-2**32+1)
    data1 = ntt_cpu.gnum(data1)
    return ntt_cpu.gnum_to_u64(data1 * coeff)


def test_prepare_for_mul(thread):
    check_func(
        thread, ntt.prepare_for_mul(), ref_prepare_for_mul,
        'ff_number', ['ff_number'])


def test_prepare_for_mul_cpu():
    array = get_test_array(1024, 'ff_number')
    res = ntt.prepare_for_mul_cpu(array)
    ref = ref_prepare_for_mul(array)
    assert (res == ref).all()


def ref_pow(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    return ntt_cpu.gnum_to_u64(data1**data2)


def test_pow(thread):
    exp_dtype = numpy.uint32
    check_func(thread, ntt.pow(exp_dtype), ref_pow, 'ff_number', ['ff_number', exp_dtype])


def ref_inv_pow2(data1):
    return ntt_cpu.gnum_to_u64(ntt_cpu.gnum(1) / ntt_cpu.gnum(2)**data1)


def test_inv_pow2(thread):
    exp_dtype = numpy.uint32
    check_func(
        thread, ntt.inv_pow2(exp_dtype), ref_inv_pow2, 'ff_number', [exp_dtype],
        ranges=[(1, 33)], test_values=[(1,), (32,)])


def ref_lsh(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(2)**data2
    return ntt_cpu.gnum_to_u64(data1 * data2)


@pytest.mark.parametrize("exp_range", [32, 64, 96, 128, 160, 192])
def test_lsh(thread, exp_range, method):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    exp_dtype = numpy.uint32
    check_func(
        thread, ntt.lsh(exp_range, exp_dtype, method=method),
        ref_lsh, 'ff_number', ['ff_number', exp_dtype],
        ranges=[None, (exp_range - 32, exp_range)],
        test_values=[
            (11509900421665959066, exp_range - 1)
        ])


def check_func_performance(
        tag, thread, func_module, reference_func, output_type, input_types, ranges=None,
        heavy_performance_load=False):

    N = 1024 * (1024 if heavy_performance_load else 32)
    repetitions = 100000

    test = get_func_kernel(thread, func_module, output_type, input_types)
    perf_test = get_func_kernel(
        thread, func_module, output_type, input_types, repetitions=repetitions)

    arrays = [
        get_test_array(N, tp, val_range=ranges[i] if ranges is not None else None)
        for i, tp in enumerate(input_types)]

    arrays_dev = [thread.to_device(arr) for arr in arrays]
    dest_dev = thread.array(N, tp_dtype(output_type))

    # Sanity check
    test(dest_dev, *arrays_dev, global_size=N)
    assert (dest_dev.get() == reference_func(*arrays)).all()

    # Performance check
    times = []

    for j in range(10):
        thread.synchronize()
        t1 = time.time()
        perf_test(dest_dev, *arrays_dev, global_size=N)
        thread.synchronize()
        t2 = time.time()
        times.append(t2 - t1)

    times = numpy.array(times)

    times /= repetitions
    times /= N
    times *= 1e12

    print()
    print(
        "{backend}: {tag} --- min: {min:.4f}, mean: {mean:.4f}, std: {std:.4f}".format(
            tag=tag,
            min=times.min(), mean=times.mean(), std=times.std(),
            backend='cuda' if thread.api.get_id() == cluda.cuda_id() else 'ocl '))


@pytest.mark.perf
def test_add_perf(thread, method, heavy_performance_load):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    if method == "c_from_asm":
        pytest.skip()

    check_func_performance(
        "add(), " + method,
        thread, ntt.add(method=method), ref_add, 'ff_number', ['ff_number', 'ff_number'],
        heavy_performance_load=heavy_performance_load)


@pytest.mark.perf
def test_sub_perf(thread, method, heavy_performance_load):
    if method == "cuda_asm" and thread.api.get_id() != cluda.cuda_id():
        pytest.skip()
    if method == "c_from_asm":
        pytest.skip()

    check_func_performance(
        "sub(), " + method,
        thread, ntt.sub(method=method), ref_sub, 'ff_number', ['ff_number', 'ff_number'],
        heavy_performance_load=heavy_performance_load)
