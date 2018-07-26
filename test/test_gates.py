import time
import numpy

import pytest
import numpy

from reikna.cluda import cuda_id

from tfhe import *


@pytest.fixture(scope='module')
def key_pair(thread):
    rng = numpy.random.RandomState()
    secret_key, cloud_key = tfhe_key_pair(thread, rng)
    return secret_key, cloud_key


@pytest.fixture(scope='module', params=[False, True], ids=['bs_loop', 'bs_kernel'])
def single_kernel_bootstrap(request):
    return request.param


def get_plaintexts(rng, num, size=32):
    return [rng.randint(0, 2, size=size).astype(numpy.bool) for i in range(num)]


def check_gate(
        thread, key_pair, perf_params, num_arguments, tfhe_func, reference_func,
        size=32, performance_test=False):

    secret_key, cloud_key = key_pair
    rng = numpy.random.RandomState()

    plaintexts = get_plaintexts(rng, num_arguments, size=size)
    ciphertexts = [tfhe_encrypt(thread, rng, secret_key, plaintext) for plaintext in plaintexts]

    reference = reference_func(plaintexts)

    params = tfhe_parameters(cloud_key)
    answer = empty_ciphertext(thread, params, (size,))

    if performance_test:

        # warm-up
        tfhe_func(thread, cloud_key, answer, *ciphertexts, perf_params)
        thread.synchronize()

        # test
        times = []
        for i in range(10):
            t_start = time.time()
            tfhe_func(thread, cloud_key, answer, *ciphertexts, perf_params)
            thread.synchronize()
            times.append(time.time() - t_start)
        times = numpy.array(times)

    else:
        tfhe_func(thread, cloud_key, answer, *ciphertexts, perf_params)
        times = None

    answer_bits = tfhe_decrypt(thread, secret_key, answer)

    assert (answer_bits == reference).all()

    return times


def test_transform_type(thread, transform_type):
    rng = numpy.random.RandomState()
    perf_params = performance_parameters()
    key_pair = tfhe_key_pair(thread, rng, transform_type=transform_type)
    check_gate(thread, key_pair, perf_params, 2, tfhe_gate_NAND_, nand_ref)


@pytest.mark.parametrize('tlwe_mask_size', [1, 2], ids=['mask_size=1', 'mask_size=2'])
def test_tlwe_mask_size(thread, tlwe_mask_size):
    rng = numpy.random.RandomState()
    perf_params = performance_parameters(single_kernel_bootstrap=False)
    key_pair = tfhe_key_pair(thread, rng, tlwe_mask_size=tlwe_mask_size)
    check_gate(thread, key_pair, perf_params, 2, tfhe_gate_NAND_, nand_ref)


def test_single_kernel_bs_with_ks(thread, key_pair, single_kernel_bootstrap):
    # Test a gate that employs a bootstrap with keyswitch
    perf_params = performance_parameters(single_kernel_bootstrap=single_kernel_bootstrap)
    check_gate(thread, key_pair, perf_params, 2, tfhe_gate_NAND_, nand_ref)


def test_single_kernel_bs(thread, key_pair, single_kernel_bootstrap):
    # Test a gate that employs separate calls to bootstrap and keyswitch
    perf_params = performance_parameters(single_kernel_bootstrap=single_kernel_bootstrap)
    check_gate(thread, key_pair, perf_params, 3, tfhe_gate_MUX_, mux_ref)


def mux_ref(plaintexts):
    assert len(plaintexts) == 3
    return plaintexts[0] * plaintexts[1] + numpy.logical_not(plaintexts[0]) * plaintexts[2]


def nand_ref(plaintexts):
    assert len(plaintexts) == 2
    return numpy.logical_not(numpy.logical_and(plaintexts[0], plaintexts[1]))


def test_mux_gate(thread, key_pair):
    perf_params = performance_parameters()
    check_gate(thread, key_pair, perf_params, 3, tfhe_gate_MUX_, mux_ref)


def test_nand_gate(thread, key_pair):
    perf_params = performance_parameters()
    check_gate(thread, key_pair, perf_params, 2, tfhe_gate_NAND_, nand_ref)


def check_performance(thread, key_pair, perf_params, size):
    # Assuming that the time taken by the gate has the form
    #   t = size * speed + overhead
    # Then, for two results t(size1), t(size2):
    #   speed = (t(size1) - t(size2)) / (size1 - size2)
    #   overhead = (t(size1) * size2 - t(size2) * size1) / (size2 - size1)

    size1 = size
    size2 = size // 2

    times1 = check_gate(
        thread, key_pair, perf_params, 2, tfhe_gate_NAND_, nand_ref,
        size=size1, performance_test=True)
    times2 = check_gate(
        thread, key_pair, perf_params, 2, tfhe_gate_NAND_, nand_ref,
        size=size2, performance_test=True)

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
def test_single_kernel_bs_performance(
        thread, transform_type, single_kernel_bootstrap, heavy_performance_load):

    size = 4096 if heavy_performance_load else 64

    perf_params = performance_parameters(single_kernel_bootstrap=single_kernel_bootstrap)
    rng = numpy.random.RandomState()
    key_pair = tfhe_key_pair(thread, rng, transform_type=transform_type)
    results = check_performance(thread, key_pair, perf_params, size=size)
    print()
    print(check_performance_str(results))


@pytest.mark.perf
@pytest.mark.parametrize('use_constant_memory', [False, True], ids=['global_mem', 'constant_mem'])
def test_constant_mem_performance(
        thread, transform_type, single_kernel_bootstrap, heavy_performance_load,
        use_constant_memory):

    size = 4096 if heavy_performance_load else 64

    perf_params = performance_parameters(
        single_kernel_bootstrap=single_kernel_bootstrap,
        use_constant_memory=use_constant_memory)
    rng = numpy.random.RandomState()
    key_pair = tfhe_key_pair(thread, rng, transform_type=transform_type)
    results = check_performance(thread, key_pair, perf_params, size=size)
    print()
    print(check_performance_str(results))


@pytest.mark.perf
@pytest.mark.parametrize(
    'transforms_per_block', [1, 2, 3, 4], ids=['tpb=1', 'tpb=2', 'tpb=3', 'tpb=4'])
def test_transforms_per_block_performance(
        thread, transform_type, heavy_performance_load, transforms_per_block):

    size = 4096 if heavy_performance_load else 64

    perf_params = performance_parameters(
        single_kernel_bootstrap=False,
        transforms_per_block=transforms_per_block)
    rng = numpy.random.RandomState()
    key_pair = tfhe_key_pair(thread, rng, transform_type=transform_type)
    results = check_performance(thread, key_pair, perf_params, size=size)
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

    perf_params = performance_parameters(
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_base_method=ntt_base_method)
    rng = numpy.random.RandomState()
    key_pair = tfhe_key_pair(thread, rng, transform_type='NTT')
    results = check_performance(thread, key_pair, perf_params, size=size)
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

    perf_params = performance_parameters(
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_mul_method=ntt_mul_method)
    rng = numpy.random.RandomState()
    key_pair = tfhe_key_pair(thread, rng, transform_type='NTT')
    results = check_performance(thread, key_pair, perf_params, size=size)
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

    perf_params = performance_parameters(
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_lsh_method=ntt_lsh_method)
    rng = numpy.random.RandomState()
    key_pair = tfhe_key_pair(thread, rng, transform_type='NTT')
    results = check_performance(thread, key_pair, perf_params, size=size)
    print()
    print(check_performance_str(results))
