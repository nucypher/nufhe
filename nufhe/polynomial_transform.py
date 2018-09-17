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

from reikna.helpers import min_blocks

from . import polynomial_transform_fft
from . import polynomial_transform_ntt


def get_transform(transform_type):
    if transform_type == 'FFT':
        return polynomial_transform_fft
    elif transform_type == 'NTT':
        return polynomial_transform_ntt


def max_supported_transforms_per_block(device_params, transform_type):
    reqs = get_transform(transform_type).transform_module_requirements()
    return min_blocks(device_params.max_work_group_size, reqs['threads_per_transform'])


def transform_supported(device_params, transform_type):
    # FFT requires double precision, otherwise the polynomial multiplication in Fourier space
    # won't have enough bits for its results.
    return device_params.supports_dtype(numpy.complex128) or not transform_type == 'FFT'
