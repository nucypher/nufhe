import pytest
import numpy

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


def check_gate(thread, key_pair, perf_params, num_arguments, tfhe_func, reference_func):

    secret_key, cloud_key = key_pair
    rng = numpy.random.RandomState()

    size = 32

    plaintexts = get_plaintexts(rng, num_arguments, size=size)
    ciphertexts = [tfhe_encrypt(thread, rng, secret_key, plaintext) for plaintext in plaintexts]

    reference = reference_func(plaintexts)

    params = tfhe_parameters(cloud_key)
    answer = empty_ciphertext(thread, params, (size,))
    tfhe_func(thread, cloud_key, answer, *ciphertexts, perf_params)

    answer_bits = tfhe_decrypt(thread, secret_key, answer)

    assert (answer_bits == reference).all()


def test_transform_type(thread, transform_type):
    rng = numpy.random.RandomState()
    perf_params = performance_parameters()
    key_pair = tfhe_key_pair(thread, rng, transform_type=transform_type)
    check_gate(thread, key_pair, perf_params, 2, tfhe_gate_NAND_, nand_ref)


@pytest.mark.parametrize('tlwe_mask_size', [1, 2], ids=['mask_size=1', 'mask_size=2'])
def test_tlwe_mask_size(thread, tlwe_mask_size):
    rng = numpy.random.RandomState()
    perf_params = performance_parameters()
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

