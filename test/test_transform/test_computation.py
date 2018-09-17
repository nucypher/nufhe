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
import itertools
import numpy
import pytest

from reikna.cluda import ocl_id, cuda_id

from nufhe.transform import fft512, ntt1024, Transform
import nufhe.transform.fft as tr_fft
import nufhe.transform.ntt as tr_ntt
from nufhe.polynomial_transform import max_supported_transforms_per_block, transform_supported

from utils import get_test_array


@pytest.mark.parametrize('inverse', [False, True], ids=['forward', 'inverse'])
@pytest.mark.parametrize('i32_conversion', [False, True], ids=['no_conversion', 'poly_conversion'])
@pytest.mark.parametrize('constant_memory', [False, True], ids=['global_mem', 'constant_mem'])
def test_transform_correctness(thread, transform_type, inverse, i32_conversion, constant_memory):

    if not transform_supported(thread.device_params, transform_type):
        pytest.skip()

    batch_shape = (128,)

    if transform_type == 'FFT':
        transform = fft512(use_constant_memory=constant_memory)
        transform_ref = tr_fft.fft_transform_ref
    else:
        transform = ntt1024(use_constant_memory=constant_memory)
        transform_ref = tr_ntt.ntt_transform_ref

    comp = Transform(
        transform, batch_shape,
        inverse=inverse, i32_conversion=i32_conversion, transforms_per_block=1,
        ).compile(thread)

    a = get_test_array(comp.parameter.input.shape, comp.parameter.input.dtype)

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(comp.parameter.output)

    comp(res_dev, a_dev)
    res_test = res_dev.get()

    res_ref = transform_ref(a, inverse=inverse, i32_conversion=i32_conversion)

    if numpy.issubdtype(res_dev.dtype, numpy.integer):
        assert (res_test == res_ref).all()
    else:
        assert numpy.allclose(res_test, res_ref)


def poly_mul_ref(p1, p2):
    N = p1.shape[-1]

    result = numpy.empty_like(p1)

    for i in range(N):
        result[:,i] = (p1[:,:i+1] * p2[:,i::-1]).sum(1) - (p1[:,i+1:] * p2[:,:i:-1]).sum(1)

    return result


def test_polynomial_multiplication(thread, transform_type):
    if not transform_supported(thread.device_params, transform_type):
        pytest.skip()

    batch_shape = (10,)

    if transform_type == 'FFT':
        transform = fft512()
        transform_ref = tr_fft.fft_transform_ref
        tr_mul_ref = tr_fft.fft_transformed_mul_ref
    else:
        transform = ntt1024()
        transform_ref = tr_ntt.ntt_transform_ref
        tr_mul_ref = tr_ntt.ntt_transformed_mul_ref

    tr_forward = Transform(
        transform, batch_shape,
        inverse=False, i32_conversion=True, transforms_per_block=1,
        ).compile(thread)
    tr_inverse = Transform(
        transform, batch_shape,
        inverse=True, i32_conversion=True, transforms_per_block=1,
        ).compile(thread)

    a = numpy.random.randint(-2**31, 2**31, size=batch_shape + (1024,), dtype=numpy.int32)
    b = numpy.random.randint(-1000, 1000, size=batch_shape + (1024,), dtype=numpy.int32)

    a_dev = thread.to_device(a)
    b_dev = thread.to_device(b)
    a_tr_dev = thread.empty_like(tr_forward.parameter.output)
    b_tr_dev = thread.empty_like(tr_forward.parameter.output)
    res_dev = thread.empty_like(tr_inverse.parameter.output)

    tr_forward(a_tr_dev, a_dev)
    tr_forward(b_tr_dev, b_dev)
    res_tr = tr_mul_ref(a_tr_dev.get(), b_tr_dev.get())
    res_tr_dev = thread.to_device(res_tr)
    tr_inverse(res_dev, res_tr_dev)

    res_test = res_dev.get()
    res_ref = poly_mul_ref(a, b)

    assert (res_test == res_ref).all()


def get_times(thread, comp, out_arr, in_arr, attempts=10):
    # Test performance
    times = []

    for j in range(attempts):
        thread.synchronize()
        t1 = time.time()
        comp(out_arr, in_arr)
        thread.synchronize()
        t2 = time.time()
        times.append(t2 - t1)

    times = numpy.array(times)

    return times, "min: {min:.4f}, mean: {mean:.4f}, std: {std:.4f}".format(
        min=times.min(), mean=times.mean(), std=times.std())


@pytest.mark.perf
@pytest.mark.parametrize('transforms_per_block', [1, 2, 4])
@pytest.mark.parametrize('constant_memory', [False, True], ids=['global_mem', 'constant_mem'])
def test_ntt_performance(thread, transforms_per_block, constant_memory, heavy_performance_load):

    if not transform_supported(thread.device_params, 'NTT'):
        pytest.skip()

    if transforms_per_block > max_supported_transforms_per_block(thread.device_params, 'NTT'):
        pytest.skip()

    is_cuda = thread.api.get_id() == cuda_id()

    methods = list(itertools.product(
        ['cuda_asm', 'c'], # base method
        ['cuda_asm', 'c_from_asm', 'c'], # mul method
        ['cuda_asm', 'c_from_asm', 'c'] # lsh method
        ))

    if not is_cuda:
        # filter out all usage of CUDA asm if we're on OpenCL
        methods = [ms for ms in methods if 'cuda_asm' not in ms]

    batch_shape = (2**14,)
    a = get_test_array(batch_shape + (1024,), "ff_number")

    kernel_repetitions = 100 if heavy_performance_load else 5

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(a_dev)

    # TODO: compute a reference NTT when it's fast enough on CPU
    #res_ref = tr_ntt.ntt_transform_ref(a)

    print()
    min_times = []
    for base_method, mul_method, lsh_method in methods:

        transform = ntt1024(
            base_method=base_method, mul_method=mul_method, lsh_method=lsh_method,
            use_constant_memory=constant_memory)

        ntt_comp = Transform(
            transform, batch_shape, transforms_per_block=transforms_per_block,
            ).compile(thread)
        ntt_comp_repeated = Transform(
            transform, batch_shape, transforms_per_block=transforms_per_block,
            kernel_repetitions=kernel_repetitions).compile(thread)

        # TODO: compute a reference NTT when it's fast enough on CPU
        # Quick check of correctness
        #ntt_comp(res_dev, a_dev)
        #res_test = res_dev.get()
        #assert (res_test == res_ref).all()

        # Test performance
        times, times_str = get_times(thread, ntt_comp_repeated, res_dev, a_dev)
        print("  base: {bm}, mul: {mm}, lsh: {lm}".format(
            bm=base_method, mm=mul_method, lm=lsh_method))
        print("  {backend}, {trnum} per block, test --- {times}".format(
            times=times_str,
            backend='cuda' if is_cuda else 'ocl ',
            trnum=transforms_per_block))

        min_times.append((times.min(), base_method, mul_method, lsh_method))

    best = min(min_times, key=lambda t: t[0])
    time_best, base_method, mul_method, lsh_method = best
    print("Best time: {tb:.4f} for [base: {bm}, mul: {mm}, lsh: {lm}]".format(
        tb=time_best, bm=base_method, mm=mul_method, lm=lsh_method
        ))


@pytest.mark.perf
@pytest.mark.parametrize('transforms_per_block', [1, 2, 3, 4])
@pytest.mark.parametrize('constant_memory', [False, True], ids=['global_mem', 'constant_mem'])
def test_fft_performance(thread, transforms_per_block, constant_memory, heavy_performance_load):

    if not transform_supported(thread.device_params, 'FFT'):
        pytest.skip()

    if transforms_per_block > max_supported_transforms_per_block(thread.device_params, 'FFT'):
        pytest.skip()

    is_cuda = thread.api.get_id() == cuda_id()

    batch_shape = (2**14,)
    a = get_test_array(batch_shape + (512,), numpy.complex128)

    kernel_repetitions = 100 if heavy_performance_load else 5

    a_dev = thread.to_device(a)
    res_dev = thread.empty_like(a_dev)

    res_ref = tr_fft.fft_transform_ref(a)

    transform = fft512(use_constant_memory=constant_memory)

    fft_comp = Transform(
        transform, batch_shape, transforms_per_block=transforms_per_block,
        ).compile(thread)
    fft_comp_repeated = Transform(
        transform, batch_shape, transforms_per_block=transforms_per_block,
        kernel_repetitions=kernel_repetitions).compile(thread)

    # Quick check of correctness
    fft_comp(res_dev, a_dev)
    res_test = res_dev.get()
    assert numpy.allclose(res_test, res_ref)

    # Test performance
    times, times_str = get_times(thread, fft_comp_repeated, res_dev, a_dev)
    print("\n{backend}, {trnum} per block, test --- {times}".format(
        times=times_str,
        backend='cuda' if is_cuda else 'ocl ',
        trnum=transforms_per_block))
