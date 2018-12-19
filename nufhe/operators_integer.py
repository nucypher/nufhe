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

from .api_low_level import empty_ciphertext
from .performance import PerformanceParameters, PerformanceParametersForDevice
from .gates import (
    gate_constant,
    gate_xnor,
    gate_mux,
    )


def _uint_to_bits(x, bitsize):
    return numpy.array([((x >> i) & 1 != 0) for i in reversed(range(bitsize))])


def _bits_to_uint(bits, dtype):
    int_answer = 0
    for i in range(bits.size):
        int_answer = int_answer | (bits[i] << (bits.size - i - 1))
    return dtype(int_answer)


def uintarray_to_bitarray(xs, itemsize=None):
    if itemsize is None:
        itemsize = xs.itemsize * 8
    assert numpy.issubdtype(xs.dtype, numpy.unsignedinteger)
    res = numpy.vstack(_uint_to_bits(x, itemsize) for x in xs.flatten())
    return res.reshape(xs.shape + (itemsize,))


def bitarray_to_uintarray(xs):

    itemsize = xs.shape[-1]
    dtype = {
        8: numpy.uint8,
        16: numpy.uint16,
        32: numpy.uint32,
        64: numpy.uint64}[itemsize]

    ints = []
    flat_xs = xs.reshape(numpy.prod(xs.shape[:-1]), xs.shape[-1])
    for j in range(flat_xs.shape[0]):
        ints.append(_bits_to_uint(flat_xs[j], dtype))
    return numpy.array(ints).reshape(xs.shape[:-1])


def uint_min(thread, cloud_key, answer, a, b, perf_params: PerformanceParametersForDevice=None):

    if perf_params is None:
        perf_params = PerformanceParameters(bk.params).for_device(thread.device_params)

    params = cloud_key.params

    itemsize = answer.shape_info.shape[-1]

    tmp1 = empty_ciphertext(thread, params, a.shape_info.shape[:-1] + (1,))
    tmp2 = empty_ciphertext(thread, params, a.shape_info.shape[:-1] + (1,))

    # initialize the carry to 0
    gate_constant(thread, cloud_key, tmp1, False)

    # Compare i-th bits in turn starting from the end (assuming big-endian order).
    # Store the result in tmp2, and use tmp1 as an accumulator.
    # Elementary full comparator gate that is used to compare the i-th bit.
    #   input: ai and bi the i-th bit of a and b
    #          lsb_carry: the result of the comparison on the lowest bits
    #   algo: if (a==b) return lsb_carry else return b
    for i in reversed(range(itemsize)):
        a_slice = a[:,i:i+1]
        b_slice = b[:,i:i+1]

        # tmp2 = (a_bit == b_bit)
        gate_xnor(thread, cloud_key, tmp2, a_slice, b_slice, perf_params=perf_params)
        # tmp1 = tmp2 ? tmp1 : a_bit
        gate_mux(thread, cloud_key, tmp1, tmp2, tmp1, a_slice, perf_params=perf_params)

    # tmp1 is the result of the comparaison: 0 if a is smaller, 1 if b is smaller
    gate_mux(thread, cloud_key, answer, tmp1, b, a, perf_params=perf_params)
