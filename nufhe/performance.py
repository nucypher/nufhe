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

from collections import namedtuple
from reikna.helpers import min_blocks
from reikna.cluda import cuda_id


class PerformanceParameters:

    def __init__(
            self,
            nufhe_params: 'NuFHEParameters',
            ntt_base_method=None,
            ntt_mul_method=None,
            ntt_lsh_method=None,
            use_constant_memory_multi_iter=None,
            use_constant_memory_single_iter=None,
            transforms_per_block=None,
            single_kernel_bootstrap=None,
            low_end_device=None):

        assert ntt_base_method in (None, 'cuda_asm', 'c')
        assert ntt_mul_method in (None, 'cuda_asm', 'c_from_asm', 'c')
        assert ntt_lsh_method in (None, 'cuda_asm', 'c_from_asm', 'c')

        self.nufhe_params = nufhe_params

        self.ntt_base_method = ntt_base_method
        self.ntt_mul_method = ntt_mul_method
        self.ntt_lsh_method = ntt_lsh_method
        self.use_constant_memory_multi_iter = use_constant_memory_multi_iter
        self.use_constant_memory_single_iter = use_constant_memory_single_iter
        self.transforms_per_block = transforms_per_block
        self.single_kernel_bootstrap = single_kernel_bootstrap
        self.low_end_device = low_end_device

    def for_device(self, device_params):
        return PerformanceParametersForDevice(self, device_params)

    def __hash__(self):
        return hash((
            self.__class__,
            self.nufhe_params,
            self.ntt_base_method,
            self.ntt_mul_method,
            self.ntt_lsh_method,
            self.use_constant_memory_multi_iter,
            self.use_constant_memory_single_iter,
            self.transforms_per_block,
            self.single_kernel_bootstrap,
            ))


class PerformanceParametersForDevice:

    def __init__(self, perf_params: PerformanceParameters, device_params):

        low_end_device = perf_params.low_end_device
        if low_end_device is None:
            # TODO: an arbitrary distinction, need to test on some devices close to it.
            low_end_device = device_params.compute_units < 20

        is_cuda = device_params.api_id == cuda_id()

        transform_type = perf_params.nufhe_params.tgsw_params.tlwe_params.transform_type
        mask_size = perf_params.nufhe_params.tgsw_params.tlwe_params.mask_size
        decomp_length = perf_params.nufhe_params.tgsw_params.decomp_length

        use_constant_memory_multi_iter = perf_params.use_constant_memory_multi_iter
        if use_constant_memory_multi_iter is None:
            use_constant_memory_multi_iter = not low_end_device

        use_constant_memory_single_iter = perf_params.use_constant_memory_single_iter
        if use_constant_memory_single_iter is None:
            use_constant_memory_single_iter = False

        transforms_per_block = perf_params.transforms_per_block
        if transforms_per_block is None:
            if low_end_device:
                transforms_per_block = 1
            else:
                transforms_per_block = 4 if transform_type == 'NTT' else 2

        # Avoiding circular reference
        from .polynomial_transform import get_transform
        reqs = get_transform(transform_type).transform_module_requirements()
        threads_per_transform = reqs['threads_per_transform']

        max_work_group_size = device_params.max_work_group_size
        max_transforms_per_block = min_blocks(max_work_group_size, threads_per_transform)
        transforms_per_block = min(transforms_per_block, max_transforms_per_block)

        # Avoiding circular reference
        from .blind_rotate import single_kernel_bootstrap_supported

        single_kernel_bootstrap = perf_params.single_kernel_bootstrap
        skbs_supported = single_kernel_bootstrap_supported(perf_params.nufhe_params, device_params)

        if single_kernel_bootstrap is None:
            # If both encryption parameters and device capabilities allow it,
            # single kernel bootstrap is the optimal choice.
            single_kernel_bootstrap = not low_end_device and skbs_supported
        elif single_kernel_bootstrap and not skbs_supported:
            single_kernel_bootstrap_supported(
                perf_params.nufhe_params, device_params, raise_exception=True)

        ntt_base_method = perf_params.ntt_base_method
        ntt_mul_method = perf_params.ntt_mul_method
        ntt_lsh_method = perf_params.ntt_lsh_method
        if low_end_device:
            skbs = single_kernel_bootstrap
            if ntt_base_method is None:
                ntt_base_method = ('c' if skbs else 'cuda_asm') if is_cuda else 'c'
            if ntt_mul_method is None:
                ntt_mul_method = ('c_from_asm' if skbs else 'cuda_asm') if is_cuda else 'c'
            if ntt_lsh_method is None:
                ntt_lsh_method = 'cuda_asm' if is_cuda else 'c'
        else:
            if ntt_base_method is None:
                ntt_base_method = 'cuda_asm' if is_cuda else 'c'
            if ntt_mul_method is None:
                ntt_mul_method = 'cuda_asm' if is_cuda else 'c'
            if ntt_lsh_method is None:
                ntt_lsh_method = 'cuda_asm' if is_cuda else 'c'

        self.ntt_base_method = ntt_base_method
        self.ntt_mul_method = ntt_mul_method
        self.ntt_lsh_method = ntt_lsh_method
        self.use_constant_memory_multi_iter = use_constant_memory_multi_iter
        self.use_constant_memory_single_iter = use_constant_memory_single_iter
        self.transforms_per_block = transforms_per_block
        self.single_kernel_bootstrap = single_kernel_bootstrap

    def __hash__(self):
        return hash((
            self.__class__,
            self.ntt_base_method,
            self.ntt_mul_method,
            self.ntt_lsh_method,
            self.use_constant_memory_multi_iter,
            self.use_constant_memory_single_iter,
            self.transforms_per_block,
            self.single_kernel_bootstrap,
            ))
