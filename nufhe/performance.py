from collections import namedtuple
from reikna.cluda import cuda_id


PerformanceParameters = namedtuple(
    'PerformanceParameters',
    [
        'single_kernel_bootstrap',
        'ntt_base_method',
        'ntt_mul_method',
        'ntt_lsh_method',
        'use_constant_memory_multi_iter',
        'use_constant_memory_single_iter',
        'transforms_per_block'
    ])


def performance_parameters(
        nufhe_params=None,
        ntt_base_method=None,
        ntt_mul_method=None,
        ntt_lsh_method=None,
        use_constant_memory_multi_iter=True,
        use_constant_memory_single_iter=False,
        transforms_per_block=None,
        single_kernel_bootstrap=None):

    assert ntt_base_method in (None, 'cuda_asm', 'c')
    assert ntt_mul_method in (None, 'cuda_asm', 'c_from_asm', 'c')
    assert ntt_lsh_method in (None, 'cuda_asm', 'c_from_asm', 'c')

    if nufhe_params is not None:
        mask_size = nufhe_params.tgsw_params.tlwe_params.mask_size
        decomp_length = nufhe_params.tgsw_params.decomp_length
        single_kernel_bootstrap_possible = mask_size == 1 and decomp_length == 2
    else:
        single_kernel_bootstrap_possible = False

    if single_kernel_bootstrap is None:
        single_kernel_bootstrap = single_kernel_bootstrap_possible
    elif single_kernel_bootstrap:
        if nufhe_params is None:
            raise ValueError(
                "The `nufhe_params` option must be specified to enable single-kernel bootstrap")
        elif not single_kernel_bootstrap_possible:
            raise ValueError(
                "Single-kernel bootstrap is only supported for mask_size=1 and decomp_length=2")

    if transforms_per_block is None:
        if nufhe_params is not None:
            transform_type = nufhe_params.tgsw_params.tlwe_params.transform_type
            transforms_per_block = 4 if transform_type == 'NTT' else 2
        else:
            transforms_per_block = 4
    else:
        assert transforms_per_block >= 1 and transforms_per_block <= 4

    return PerformanceParameters(
        single_kernel_bootstrap=single_kernel_bootstrap,
        ntt_base_method=ntt_base_method,
        ntt_mul_method=ntt_mul_method,
        ntt_lsh_method=ntt_lsh_method,
        use_constant_memory_multi_iter=use_constant_memory_multi_iter,
        use_constant_memory_single_iter=use_constant_memory_single_iter,
        transforms_per_block=transforms_per_block,
        )


def performance_parameters_for_device(perf_params, device_params):

    is_cuda = device_params.api_id == cuda_id()

    ntt_base_method = perf_params.ntt_base_method
    ntt_mul_method = perf_params.ntt_mul_method
    ntt_lsh_method = perf_params.ntt_lsh_method

    if ntt_base_method is None:
        ntt_base_method = 'cuda_asm' if is_cuda else 'c'
    if ntt_mul_method is None:
        ntt_mul_method = 'cuda_asm' if is_cuda else 'c'
    if ntt_lsh_method is None:
        ntt_lsh_method = 'cuda_asm' if is_cuda else 'c'

    pdict = perf_params._asdict()
    pdict.update(
        ntt_base_method=ntt_base_method,
        ntt_mul_method=ntt_mul_method,
        ntt_lsh_method=ntt_lsh_method,
        )

    return PerformanceParameters(**pdict)
