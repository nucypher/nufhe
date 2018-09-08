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
import pytest

import nufhe.transform.ntt_cpu as ntt_cpu


def tp_dtype(tp):
    if isinstance(tp, str) and tp == 'ff_number':
        return numpy.uint64
    else:
        return tp


def tp_limits(tp):
    if isinstance(tp, str) and tp == 'ff_number':
        return 0, ntt_cpu.GaloisNumber.modulus
    elif numpy.issubdtype(tp, numpy.integer):
        ii = numpy.iinfo(tp)
        return ii.min, ii.max + 1
    else:
        return -10, 10


def get_test_array(shape, tp, val_range=None):
    dtype = tp_dtype(tp)
    if val_range is None:
        nmin, nmax = tp_limits(tp)
    else:
        nmin, nmax = val_range

    if numpy.issubdtype(dtype, numpy.integer):
        return numpy.random.randint(nmin, nmax, dtype=dtype, size=shape)
    elif numpy.issubdtype(dtype, numpy.floating):
        return numpy.random.uniform(nmin, nmax, size=shape).astype(dtype)
    elif numpy.issubdtype(dtype, numpy.complexfloating):
        return (
            numpy.random.uniform(nmin, nmax, size=shape)
            + 1j * numpy.random.uniform(nmin, nmax, size=shape)).astype(dtype)
    else:
        raise NotImplementedError(dtype)


def supports_transform(thread, transform_type):
    # FFT required double precision, otherwise the polynomial multiplication in Fourier space
    # won't have enough bits for its results.
    return thread.device_params.supports_dtype(numpy.complex128) or not transform_type == 'FFT'
