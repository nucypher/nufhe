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
from reikna.cluda import cuda_id


class PerformanceParameters:
    """
    Advanced performance settings for bootstrapping.

    For all the optional parameters below, if ``None`` is given,
    the library will attempt to select the best performing variant,
    given the available information.

    :param nufhe_params: a :py:class:`NuFHEParameters` object.
    :param ntt_base_method: ``'cuda_asm'`` or ``'c'``;
        An algorithm used in NTT for modulo addition, modulo subtraction, and modulus.
    :param ntt_mul_method: one of ``'cuda_asm'``, ``'c_from_asm'`` and ``'c'``;
        An algorithm used in NTT for modulo multiplication.
    :param ntt_lsh_method: one of ``'cuda_asm'``, ``'c_from_asm'`` and ``'c'``;
        An algorithm used in NTT for modulo bitshift.

    .. note::

        ``'cuda_asm'`` is only available for CUDA backend.
        When available, it is usually the fastest variant, or close to it.

    :param use_constant_memory_multi_iter: use constant GPU memory (as opposed to global memory)
        for precalculated coefficients in NTT/FFT in kernels where one of these transformations
        is executed multiple times per kernel call.
    :param use_constant_memory_single_iter: use constant GPU memory (as opposed to global memory)
        for precalculated coefficients in NTT/FFT in kernels where one of these transformations
        is executed once per kernel call.

    .. note::

        Using constant memory is usually beneficial on fast videocards
        if the transformation is executed many times per kernel call.

    :param transforms_per_block: a positive integer value, denoting how many separate transforms
        to execute in parallel on a single GPU multiprocessor (compute unit).

    .. note::

        On most videocards 1 to 4 transforms is supported for NTT, and 1 to 8 for FFT.
        More transforms does not necessarily mean better performance, since parallel threads
        on the same compute unit compete for resources.

        Since it is not trivial to determine the maximum in advance, if the requested number
        is greater than that, it will be dynamically reduced to the maximum possible value.

    :param single_kernel_bootstrap: if ``True``, execute bootstrap in a single kernel,
        instead of many separate kernel calls in a loop.

    .. note::

        Single kernel bootstrap is only supported for default FHE parameters,
        and needs the videocard to support a certain amount of parallel threads on a compute unit
        (256 for FFT, 512 for NTT).
        If available, it is usually significantly faster,
        partially due to lower kernel call overhead.

    :param low_end_device: if ``True``, the optimal values for low-end videocards will be picked.
        If ``None``, the decision will be made based on the number of compute units the device has.
    """

    __attributes__ = (
        'nufhe_params',
        'ntt_base_method',
        'ntt_mul_method',
        'ntt_lsh_method',
        'use_constant_memory_multi_iter',
        'use_constant_memory_single_iter',
        'transforms_per_block',
        'single_kernel_bootstrap',
        'low_end_device',
        )

    def __init__(
            self,
            nufhe_params, # TODO: type annotation here ('NuFHEParameters') triggers a Sphinx error.
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
        """
        Specialize performance parameters for the given device
        (using a Reikna ``DeviceParams`` object).

        :returns: a :py:class:`~nufhe.performance.PerformanceParametersForDevice` object.
        """
        return PerformanceParametersForDevice(self, device_params)

    def __hash__(self):
        return hash((self.__class__,) + self.__attributes__)

    def __eq__(self, other):
        return all(getattr(self, attr) == getattr(other, attr) for attr in self.__attributes__)


class PerformanceParametersForDevice:

    __attributes__ = (
        'ntt_base_method',
        'ntt_mul_method',
        'ntt_lsh_method',
        'use_constant_memory_multi_iter',
        'use_constant_memory_single_iter',
        'transforms_per_block',
        'single_kernel_bootstrap',
        )

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

        # Avoiding circular reference
        from .polynomial_transform import max_supported_transforms_per_block
        max_supported_tpb = max_supported_transforms_per_block(device_params, transform_type)

        transforms_per_block = perf_params.transforms_per_block
        if transforms_per_block is None:
            if low_end_device:
                transforms_per_block = 1
            else:
                transforms_per_block = 4 if transform_type == 'NTT' else 2
            transforms_per_block = min(transforms_per_block, max_supported_tpb)
        else:
            if transforms_per_block > max_supported_tpb:
                raise ValueError(
                    "The chosen device does not support more than " + str(max_supported_tpb) +
                    " transforms per block")

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

        if 'cuda_asm' in (ntt_base_method, ntt_mul_method, ntt_lsh_method) and not is_cuda:
            raise ValueError("'cuda_asm' is only supported for the CUDA backend")

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
        return hash((self.__class__,) + self.__attributes__)

    def __eq__(self, other):
        return all(getattr(self, attr) == getattr(other, attr) for attr in self.__attributes__)
