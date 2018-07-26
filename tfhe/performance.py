from collections import namedtuple


PerformanceParameters = namedtuple(
    'PerformanceParameters',
    [
        'single_kernel_bootstrap',
        'ntt_base_method',
        'ntt_mul_method',
        'ntt_lsh_method',
        'use_constant_memory',
        'transforms_per_block'
    ])


def performance_parameters(
        thread=None,
        ntt_base_method=None,
        ntt_mul_method=None,
        ntt_lsh_method=None,
        use_constant_memory=True,
        transforms_per_block=2,
        single_kernel_bootstrap=True):

    assert ntt_base_method in (None, 'cuda_asm', 'c')
    assert ntt_mul_method in (None, 'cuda_asm', 'c_from_asm', 'c')
    assert ntt_lsh_method in (None, 'cuda_asm', 'c_from_asm', 'c')

    if ntt_base_method is None:
        ntt_base_method = 'c'
    if ntt_mul_method is None:
        ntt_mul_method = 'c'
    if ntt_lsh_method is None:
        ntt_lsh_method = 'c'

    return PerformanceParameters(
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_base_method=ntt_base_method,
        ntt_mul_method=ntt_mul_method,
        ntt_lsh_method=ntt_lsh_method,
        use_constant_memory=use_constant_memory,
        transforms_per_block=transforms_per_block,
        )
