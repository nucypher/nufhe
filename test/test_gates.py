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

import pytest
import numpy

from reikna.cluda import cuda_id

from nufhe import *
from nufhe.operators_integer import uint_min, bitarray_to_uintarray, uintarray_to_bitarray
from nufhe.blind_rotate import single_kernel_bootstrap_supported
from nufhe.polynomial_transform import max_supported_transforms_per_block, transform_supported


@pytest.fixture(scope='module', params=[False, True], ids=['bs_loop', 'bs_kernel'])
def single_kernel_bootstrap(request):
    return request.param


def get_plaintexts(rng, num, shape=(32,)):
    return [rng.uniform_bool(shape).astype(numpy.bool) for i in range(num)]


def check_gate(
        thread, key_pair, num_arguments, nufhe_func, reference_func,
        shape=32, performance_test=False, perf_params=None):

    if not isinstance(shape, tuple):
        shape = (shape,)

    secret_key, cloud_key = key_pair

    if perf_params is None:
        perf_params = PerformanceParameters(secret_key.params).for_device(thread.device_params)

    rng = DeterministicRNG()

    plaintexts = get_plaintexts(rng, num_arguments, shape=shape)
    ciphertexts = [encrypt(thread, rng, secret_key, plaintext) for plaintext in plaintexts]

    reference = reference_func(*plaintexts)

    params = cloud_key.params
    answer = empty_ciphertext(thread, params, shape)

    if performance_test:

        # warm-up
        nufhe_func(thread, cloud_key, answer, *ciphertexts, perf_params)
        thread.synchronize()

        # test
        times = []
        for i in range(10):
            t_start = time.time()
            nufhe_func(thread, cloud_key, answer, *ciphertexts, perf_params)
            thread.synchronize()
            times.append(time.time() - t_start)
        times = numpy.array(times)

    else:
        nufhe_func(thread, cloud_key, answer, *ciphertexts, perf_params)
        times = None

    answer_bits = decrypt(thread, secret_key, answer)

    assert (answer_bits == reference).all()

    return times


def test_transform_type(thread, transform_type):
    if not transform_supported(thread.device_params, transform_type):
        pytest.skip()
    rng = DeterministicRNG()
    key_pair = make_key_pair(thread, rng, transform_type=transform_type)
    check_gate(thread, key_pair, 2, gate_nand, nand_ref)


@pytest.mark.parametrize('tlwe_mask_size', [1, 2], ids=['mask_size=1', 'mask_size=2'])
def test_tlwe_mask_size(thread, tlwe_mask_size):
    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng, tlwe_mask_size=tlwe_mask_size)
    check_gate(thread, (secret_key, cloud_key), 2, gate_nand, nand_ref)


def test_single_kernel_bs_with_ks(thread, key_pair, single_kernel_bootstrap):
    # Test a gate that employs a bootstrap with keyswitch
    secret_key, cloud_key = key_pair

    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(secret_key.params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(
        secret_key.params, single_kernel_bootstrap=single_kernel_bootstrap)
    perf_params = perf_params.for_device(thread.device_params)
    check_gate(thread, key_pair, 2, gate_nand, nand_ref, perf_params=perf_params)


def test_single_kernel_bs(thread, key_pair, single_kernel_bootstrap):
    # Test a gate that employs separate calls to bootstrap and keyswitch
    secret_key, cloud_key = key_pair

    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(secret_key.params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(
        secret_key.params, single_kernel_bootstrap=single_kernel_bootstrap)
    perf_params = perf_params.for_device(thread.device_params)
    check_gate(thread, key_pair, 3, gate_mux, mux_ref, perf_params=perf_params)


def nand_ref(a, b):
    return ~(a * b)

def or_ref(a, b):
    return a + b

def and_ref(a, b):
    return a * b

def xor_ref(a, b):
    return a ^ b

def xnor_ref(a, b):
    return ~(a ^ b)

def not_ref(a):
    return ~a

def copy_ref(a):
    return a

def nor_ref(a, b):
    return ~(a + b)

def andny_ref(a, b):
    return ~a * b

def andyn_ref(a, b):
    return a * ~b

def orny_ref(a, b):
    return ~a + b

def oryn_ref(a, b):
    return a + ~b

def mux_ref(a, b, c):
    return a * b + ~a * c

def uint_min_ref(p1, p2):
    ints1 = bitarray_to_uintarray(p1)
    ints2 = bitarray_to_uintarray(p2)
    ires = numpy.minimum(ints1, ints2)
    res = uintarray_to_bitarray(ires)
    return res


def test_nand_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_nand, nand_ref)


def test_or_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_or, or_ref)


def test_and_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_and, and_ref)


def test_xor_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_xor, xor_ref)


def test_xnor_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_xnor, xnor_ref)


def test_not_gate(thread, key_pair):
    check_gate(thread, key_pair, 1, gate_not, not_ref)


def test_copy_gate(thread, key_pair):
    check_gate(thread, key_pair, 1, gate_copy, copy_ref)


def test_nor_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_nor, nor_ref)


def test_andny_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_andny, andny_ref)


def test_andyn_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_andyn, andyn_ref)


def test_orny_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_orny, orny_ref)


def test_oryn_gate(thread, key_pair):
    check_gate(thread, key_pair, 2, gate_oryn, oryn_ref)


def test_mux_gate(thread, key_pair):
    check_gate(thread, key_pair, 3, gate_mux, mux_ref)


def test_constant_gate(thread, key_pair):
    # Not using check_gate(), because no encryption is required.

    size = 32

    secret_key, cloud_key = key_pair
    rng = DeterministicRNG()

    params = cloud_key.params
    answer = empty_ciphertext(thread, params, (size,))

    vals = get_plaintexts(rng, 1, shape=(size,))[0]

    gate_constant(thread, cloud_key, answer, vals)
    answer_bits = decrypt(thread, secret_key, answer)
    assert (answer_bits == vals).all()


def test_uint_min(thread, key_pair):
    check_gate(thread, key_pair, 2, uint_min, uint_min_ref, shape=(4, 16))


def check_performance(
        thread, key_pair, perf_params, shape, test_function=(gate_nand, nand_ref, 2)):

    # Assuming that the time taken by the gate has the form
    #   t = size * speed + overhead
    # Then, for two results t(size1), t(size2):
    #   speed = (t(size1) - t(size2)) / (size1 - size2)
    #   overhead = (t(size1) * size2 - t(size2) * size1) / (size2 - size1)

    nufhe_func, ref_func, nargs = test_function

    if isinstance(shape, tuple):
        shape1 = shape
        shape2 = (shape[0] // 2,) + shape[1:]
        size1 = numpy.prod(shape1)
        size2 = numpy.prod(shape2)
    else:
        shape1 = shape
        shape2 = shape // 2
        size1 = shape1
        size2 = shape2

    times1 = check_gate(
        thread, key_pair, nargs, nufhe_func, ref_func,
        shape=shape1, performance_test=True, perf_params=perf_params)
    times2 = check_gate(
        thread, key_pair, nargs, nufhe_func, ref_func,
        shape=shape2, performance_test=True, perf_params=perf_params)

    mean1 = times1.mean()
    err1 = times1.std() / times1.size**0.5
    mean2 = times2.mean()
    err2 = times2.std() / times2.size**0.5

    speed_overall_mean = mean1 / size1
    speed_overall_err = err1 / size1

    speed_mean = (mean1 - mean2) / (size1 - size2)
    speed_err = abs((err1 + err2) / (size1 - size2))

    overhead_mean = (mean1 * size2 - mean2 * size1) / (size2 - size1)
    overhead_err = abs((err1 * size2 + err2 * size2) / (size2 - size1))

    return dict(
        speed_overall_mean=speed_overall_mean,
        speed_overall_err=speed_overall_err,
        speed_mean=speed_mean,
        speed_err=speed_err,
        overhead_mean=overhead_mean,
        overhead_err=overhead_err)


def check_performance_str(results):
    return (
        "Overall speed: {somean:.4f} +/- {soerr:.4f} ms/bit, " +
        "scaled: {smean:.4f} +/- {serr:.4f} ms/bit, " +
        "overhead: {omean:.4f} +/- {oerr:.4f} ms").format(
        somean=results['speed_overall_mean'] * 1e3,
        soerr=results['speed_overall_err'] * 1e3,
        smean=results['speed_mean'] * 1e3,
        serr=results['speed_err'] * 1e3,
        omean=results['overhead_mean'] * 1e3,
        oerr=results['overhead_err'] * 1e3)


@pytest.mark.perf
@pytest.mark.parametrize('test_function_name', ['NAND', 'MUX', 'uint_min'])
def test_single_kernel_bs_performance(
        thread, transform_type, single_kernel_bootstrap,
        test_function_name, heavy_performance_load):

    if not transform_supported(thread.device_params, transform_type):
        pytest.skip()

    test_function = dict(
        NAND=(gate_nand, nand_ref, 2),
        MUX=(gate_mux, mux_ref, 3),
        uint_min=(uint_min, uint_min_ref, 2),
        )[test_function_name]

    if test_function_name == 'uint_min':
        shape = (128, 32) if heavy_performance_load else (4, 16)
    else:
        shape = 4096 if heavy_performance_load else 64

    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng, transform_type=transform_type)

    # TODO: instead of creating a whole key and then checking if the parameters are supported,
    # we can just create a parameter object separately.
    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(secret_key.params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(
        secret_key.params, single_kernel_bootstrap=single_kernel_bootstrap)
    perf_params = perf_params.for_device(thread.device_params)

    results = check_performance(
        thread, (secret_key, cloud_key), perf_params, shape=shape, test_function=test_function)
    print()
    print(check_performance_str(results))


@pytest.mark.perf
@pytest.mark.parametrize('use_constant_memory', [False, True], ids=['global_mem', 'constant_mem'])
def test_constant_mem_performance(
        thread, transform_type, single_kernel_bootstrap, heavy_performance_load,
        use_constant_memory):

    if not transform_supported(thread.device_params, transform_type):
        pytest.skip()

    # We want to test the effect of using constant memory on the bootstrap calculation.
    # A single-kernel bootstrap uses the `use_constant_memory_multi_iter` option,
    # and a multi-kernel bootstrap uses the `use_constant_memory_single_iter` option.
    kwds = dict(single_kernel_bootstrap=single_kernel_bootstrap)
    if single_kernel_bootstrap:
        kwds.update(dict(use_constant_memory_multi_iter=use_constant_memory))
    else:
        kwds.update(dict(use_constant_memory_single_iter=use_constant_memory))

    size = 4096 if heavy_performance_load else 64

    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng, transform_type=transform_type)

    # TODO: instead of creating a whole key and then checking if the parameters are supported,
    # we can just create a parameter object separately.
    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(secret_key.params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(secret_key.params, **kwds).for_device(thread.device_params)

    results = check_performance(thread, (secret_key, cloud_key), perf_params, shape=size)
    print()
    print(check_performance_str(results))


@pytest.mark.perf
@pytest.mark.parametrize(
    'transforms_per_block', [1, 2, 3, 4], ids=['tpb=1', 'tpb=2', 'tpb=3', 'tpb=4'])
def test_transforms_per_block_performance(
        thread, transform_type, heavy_performance_load, transforms_per_block):

    if not transform_supported(thread.device_params, transform_type):
        pytest.skip()

    max_tpb = max_supported_transforms_per_block(thread.device_params, transform_type)
    if transforms_per_block > max_tpb:
        pytest.skip()

    size = 4096 if heavy_performance_load else 64

    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng, transform_type=transform_type)

    perf_params = PerformanceParameters(
        secret_key.params,
        single_kernel_bootstrap=False,
        transforms_per_block=transforms_per_block).for_device(thread.device_params)

    results = check_performance(thread, (secret_key, cloud_key), perf_params, shape=size)
    print()
    print(check_performance_str(results))


@pytest.mark.perf
@pytest.mark.parametrize(
    'ntt_base_method', ['cuda_asm', 'c'], ids=['ntt_base=cuda_asm', 'ntt_base=c'])
def test_ntt_base_method_performance(
        thread, single_kernel_bootstrap, heavy_performance_load, ntt_base_method):

    if thread.api.get_id() != cuda_id() and ntt_base_method == 'cuda_asm':
        pytest.skip()

    size = 4096 if heavy_performance_load else 64

    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng, transform_type='NTT')

    # TODO: instead of creating a whole key and then checking if the parameters are supported,
    # we can just create a parameter object separately.
    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(secret_key.params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(
        secret_key.params,
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_base_method=ntt_base_method).for_device(thread.device_params)

    results = check_performance(thread, (secret_key, cloud_key), perf_params, shape=size)
    print()
    print(check_performance_str(results))


@pytest.mark.perf
@pytest.mark.parametrize(
    'ntt_mul_method',
    ['cuda_asm', 'c_from_asm', 'c'],
    ids=['ntt_mul=cuda_asm', 'ntt_mul=c_from_asm', 'ntt_mul=c'])
def test_ntt_mul_method_performance(
        thread, single_kernel_bootstrap, heavy_performance_load, ntt_mul_method):

    if thread.api.get_id() != cuda_id() and ntt_mul_method == 'cuda_asm':
        pytest.skip()

    size = 4096 if heavy_performance_load else 64

    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng, transform_type='NTT')

    # TODO: instead of creating a whole key and then checking if the parameters are supported,
    # we can just create a parameter object separately.
    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(secret_key.params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(
        secret_key.params,
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_mul_method=ntt_mul_method).for_device(thread.device_params)

    results = check_performance(thread, (secret_key, cloud_key), perf_params, shape=size)
    print()
    print(check_performance_str(results))


@pytest.mark.perf
@pytest.mark.parametrize(
    'ntt_lsh_method',
    ['cuda_asm', 'c_from_asm', 'c'],
    ids=['ntt_lsh=cuda_asm', 'ntt_lsh=c_from_asm', 'ntt_lsh=c'])
def test_ntt_lsh_method_performance(
        thread, single_kernel_bootstrap, heavy_performance_load, ntt_lsh_method):

    if thread.api.get_id() != cuda_id() and ntt_lsh_method == 'cuda_asm':
        pytest.skip()

    size = 4096 if heavy_performance_load else 64

    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng, transform_type='NTT')

    # TODO: instead of creating a whole key and then checking if the parameters are supported,
    # we can just create a parameter object separately.
    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(secret_key.params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(
        secret_key.params,
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_lsh_method=ntt_lsh_method).for_device(thread.device_params)

    results = check_performance(thread, (secret_key, cloud_key), perf_params, shape=size)
    print()
    print(check_performance_str(results))


def test_gate_over_view(thread, key_pair, single_kernel_bootstrap):

    secret_key, cloud_key = key_pair
    params = cloud_key.params

    if (single_kernel_bootstrap
            and not single_kernel_bootstrap_supported(params, thread.device_params)):
        pytest.skip()

    perf_params = PerformanceParameters(params, single_kernel_bootstrap=single_kernel_bootstrap)
    perf_params = perf_params.for_device(thread.device_params)

    nufhe_func = gate_nand
    reference_func = nand_ref
    num_arguments = 2

    rng = DeterministicRNG()

    shape = (5, 8)

    # FIXME: negative steps are supported as well, but the current stable PyCUDA
    # has a bug where in that case it calculates strides incorrectly.
    # It is fixed in the trunk, so we must add some negative steps here as soon as it is released.
    slices1 = (slice(3, 5), slice(1, 7, 2))
    slices2 = (slice(1, 3), slice(2, 8, 2))
    result_slices = (slice(2, 4), slice(0, 6, 2))

    plaintexts = get_plaintexts(rng, num_arguments, shape=shape)
    pt1 = plaintexts[0][slices1]
    pt2 = plaintexts[1][slices2]

    ciphertexts = [encrypt(thread, rng, secret_key, plaintext) for plaintext in plaintexts]
    ct1 = ciphertexts[0][slices1]
    ct2 = ciphertexts[1][slices2]

    reference = reference_func(pt1, pt2)

    answer = empty_ciphertext(thread, params, shape)
    answer_view = answer[result_slices]

    nufhe_func(thread, cloud_key, answer_view, ct1, ct2, perf_params=perf_params)

    answer_bits = decrypt(thread, secret_key, answer)
    answer_bits_view = answer_bits[result_slices]

    assert (answer_bits_view == reference).all()
