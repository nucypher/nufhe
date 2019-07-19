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

import reikna.helpers as helpers
from reikna.cluda import dtypes, Module

from . import arithmetic
from . import ntt_cpu as ntt_cpu


TEMPLATE = helpers.template_for(__file__)


def ntt_transform_ref(data, inverse=False, i32_conversion=False):
    N = data.shape[-1]
    data = ntt_cpu.gnum(data)
    w = ntt_cpu.root_of_unity(2 * N)
    forward_coeffs = numpy.array([w**i for i in numpy.arange(N)])

    if inverse:
        inverse_coeffs = ntt_cpu.gnum(1) / forward_coeffs
        res = ntt_cpu.ntt(data, True) * inverse_coeffs
        if i32_conversion:
            return ntt_cpu.gnum_to_i32(res)
        else:
            return ntt_cpu.gnum_to_u64(res)
    else:
        return ntt_cpu.gnum_to_u64(ntt_cpu.ntt(data * forward_coeffs, False))


def ntt_transformed_add_ref(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 + data2)


def ntt_transformed_mul_ref(data1, data2):
    data1 = ntt_cpu.gnum(data1)
    data2 = ntt_cpu.gnum(data2)
    return ntt_cpu.gnum_to_u64(data1 * data2)


def root_ref(n):
    return ntt_cpu.gnum(0xa70dc47e4cbdf43f)**(2**32 // n)


def gen_twiddle_ref():
    n = 1024

    twd = numpy.empty(n, numpy.uint64)
    twd_inv = numpy.empty(n, numpy.uint64)

    w = root_ref(n)
    for tid0 in range(2):
        for tid1 in range(8):
            for tid2 in range(8):
                cid = (tid0 << 6) + (tid1 << 3) + tid2
                for i in range(8):
                    e = (tid0 * 8 + tid1 // 4 * 4 + (tid2 % 4)) * (i * 8 + (tid1 % 4) * 2 + tid2 // 4)
                    idx = (i * n // 8) + cid

                    twd[idx] = ntt_cpu.gnum_to_u64(w**e)
                    twd_inv[idx] = ntt_cpu.gnum_to_u64(w**((n - e) % n))


    twd_sqrt = numpy.empty(n, numpy.uint64)
    twd_sqrt_inv = numpy.empty(n, numpy.uint64)

    w = root_ref(2 * n)
    n_inv = ntt_cpu.gnum(1) / ntt_cpu.gnum(2)**10

    for idx in range(1024):
        twd_sqrt[idx] = ntt_cpu.gnum_to_u64(w**idx)
        twd_sqrt_inv[idx] = ntt_cpu.gnum_to_u64(w**((2 * n - idx) % (2 * n)) * n_inv)


    return twd, twd_inv, twd_sqrt, twd_sqrt_inv


class NTT1024:

    def __init__(self, ff_elem, module, use_constant_memory):
        self.ff = ff_elem
        self.module = module
        self.use_constant_memory = use_constant_memory

        self.transform_length = 1024
        self.elem_dtype = numpy.dtype('uint64')
        self.elem_ctype = ff_elem.module

        self.polynomial_length = 1024
        self.polynomial_dtype = numpy.dtype('int32')
        self.polynomial_ctype = dtypes.ctype(self.polynomial_dtype)

        self.threads_per_transform = 128

        self.temp_dtype = numpy.dtype('uint64')
        self.temp_ctype = ff_elem.module
        self.temp_length = 1024

        twd, twd_inv, twd_sqrt, twd_sqrt_inv = gen_twiddle_ref()

        self.cdata_fw = arithmetic.prepare_for_mul_cpu(numpy.concatenate([twd, twd_sqrt]))
        self.cdata_inv = arithmetic.prepare_for_mul_cpu(numpy.concatenate([twd_inv, twd_sqrt_inv]))

        self.cdata_fw_ctype = ff_elem.module
        self.cdata_inv_ctype = ff_elem.module

    def __process_modules__(self, process):
        return NTT1024(process(self.ff), process(self.module), self.use_constant_memory)


def ntt1024(
        base_method='c', mul_method='c_from_asm', lsh_method='c', ff_elem=None,
        use_constant_memory=False):

    if ff_elem is None:
        ff_elem = arithmetic.get_ff_elem()

    base_kwds = dict(ff_elem=ff_elem, method=base_method)
    mul_prepared_kwds = dict(ff_elem=ff_elem, nested_method=base_method)
    lsh_kwds = dict(ff_elem=ff_elem, method=lsh_method, nested_method=base_method)

    module = Module(
        TEMPLATE.get_def('ntt1024'),
        render_kwds=dict(
            ff=ff_elem,
            ff_elem=ff_elem.module,
            lsh32=arithmetic.lsh(32, numpy.uint32, **lsh_kwds).module,
            lsh64=arithmetic.lsh(64, numpy.uint32, **lsh_kwds).module,
            lsh96=arithmetic.lsh(96, numpy.uint32, **lsh_kwds).module,
            lsh128=arithmetic.lsh(128, numpy.uint32, **lsh_kwds).module,
            lsh160=arithmetic.lsh(160, numpy.uint32, **lsh_kwds).module,
            lsh192=arithmetic.lsh(192, numpy.uint32, **lsh_kwds).module,
            add=arithmetic.add(**base_kwds).module,
            sub=arithmetic.sub(**base_kwds).module,
            mul_prepared=arithmetic.mul_prepared(**mul_prepared_kwds).module,
            use_constant_memory=use_constant_memory,
            ))
    return NTT1024(ff_elem, module, use_constant_memory)


def ntt1024_requirements():
    return dict(
        threads_per_transform=128,
        transform_length=1024,
        temp_length=1024,
        elem_dtype_itemsize=numpy.dtype('uint64').itemsize,
        temp_dtype_itemsize=numpy.dtype('uint64').itemsize,
        polynomial_length=1024)
