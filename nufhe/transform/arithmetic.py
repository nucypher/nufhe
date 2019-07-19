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


TEMPLATE = helpers.template_for(__file__)


class FiniteFieldElement:

    def __init__(self, module):
        self.module = module
        self.u64 = dtypes.ctype(numpy.uint64)
        self.u32 = dtypes.ctype(numpy.uint32)
        self.modulus = dtypes.c_constant(2**64 - 2**32 + 1, numpy.uint64)

    def __process_modules__(self, process):
        return FiniteFieldElement(process(self.module))


def get_ff_elem():
    module = Module(
        TEMPLATE.get_def('ff_elem_def'),
        render_kwds=dict(u64=dtypes.ctype(numpy.uint64)))
    return FiniteFieldElement(module)


class FiniteFieldAdd:

    def __init__(self, ff_elem, module):
        self.ff = ff_elem
        self.module = module

    def __process_modules__(self, process):
        return FiniteFieldAdd(process(self.ff), process(self.module))


def add(ff_elem=None, method="c"):
    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('add_def'),
        render_kwds=dict(method=method, ff=ff_elem, ff_elem=ff_elem.module))
    return FiniteFieldAdd(ff_elem, module)


class FiniteFieldSub:

    def __init__(self, ff_elem, module):
        self.ff = ff_elem
        self.module = module

    def __process_modules__(self, process):
        return FiniteFieldSub(process(self.ff), process(self.module))


def sub(ff_elem=None, method="c"):
    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('sub_def'),
        render_kwds=dict(method=method, ff=ff_elem, ff_elem=ff_elem.module))
    return FiniteFieldSub(ff_elem, module)


class FiniteFieldMod:

    def __init__(self, ff_elem, module):
        self.ff = ff_elem
        self.module = module

    def __process_modules__(self, process):
        return FiniteFieldMod(process(self.ff), process(self.module))


def mod(ff_elem=None, method="c"):
    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('mod_def'),
        render_kwds=dict(method=method, ff=ff_elem, ff_elem=ff_elem.module))
    return FiniteFieldMod(ff_elem, module)


class FiniteFieldMul:

    def __init__(self, ff_elem, module):
        self.ff = ff_elem
        self.module = module

    def __process_modules__(self, process):
        return FiniteFieldMul(process(self.ff), process(self.module))


def mul(ff_elem=None, method="c", nested_method="c"):
    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('mul_def'),
        render_kwds=dict(
            method=method,
            ff=ff_elem, ff_elem=ff_elem.module,
            add=add(ff_elem=ff_elem, method=nested_method).module, # used for "c" method
            sub=sub(ff_elem=ff_elem, method=nested_method).module,
            mod=mod(ff_elem=ff_elem, method=nested_method).module, # used for "c_from_asm" method
            ))
    return FiniteFieldMul(ff_elem, module)


class FiniteFieldMulPrepared:

    def __init__(self, ff_elem, module):
        self.ff = ff_elem
        self.module = module

    def __process_modules__(self, process):
        return FiniteFieldMulPrepared(process(self.ff), process(self.module))


def mul_prepared(ff_elem=None, method="c", nested_method="c"):
    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('mul_prepared_def'),
        render_kwds=dict(
            method=method,
            ff=ff_elem, ff_elem=ff_elem.module,
            sub=sub(ff_elem=ff_elem, method=nested_method).module,
            ))
    return FiniteFieldMulPrepared(ff_elem, module)


class FiniteFieldPrepareForMul:

    def __init__(self, ff_elem, module):
        self.ff = ff_elem
        self.module = module

    def __process_modules__(self, process):
        return FiniteFieldPrepareForMul(process(self.ff), process(self.module))


def prepare_for_mul(ff_elem=None):
    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('prepare_for_mul_def'),
        render_kwds=dict(
            ff=ff_elem, ff_elem=ff_elem.module, sub=sub(ff_elem=ff_elem).module,
            ))
    return FiniteFieldPrepareForMul(ff_elem, module)


def prepare_for_mul_cpu(x):
    """
    Converts a numpy array `x` of type `uint64` into Montgomery represenation
    for modulus `M = 2**64-2**32+1` and word size `R = 2**64`.
    That is, returns `x * R mod M`.
    """

    # This is the CPU analogue of prepare_for_mul()

    # Using the known form of modulus, we are rewriting multiplications and modulo operations
    # as shifts to make it faster.
    #
    # The idea is as follows: `let r = 2**32` (so `r**2 = R`).
    # Then `(a + b * r) * r**2 mod (r**2 - r + 1) = a * r - a - b mod (r**2 - r + 1)`.
    # `a * r - a` is always in range `[0, m)`, so we just need to check if it is
    # smaller than `b` and, in that case, add `m` (which is equivalent to subtracting `r - 1`
    # with modulo `r**2` applied automatically since we're working with `uint64`).

    x_lo = x & numpy.uint64(0xffffffff)
    x_hi = x >> 32
    y = (x_lo << 32) - x_lo
    adjust = (y < x_hi).astype(numpy.uint64)
    adjust = (adjust << 32) - adjust
    return y - x_hi - adjust


class FiniteFieldPow:

    def __init__(self, ff_elem, module, exp_dtype):
        self.ff = ff_elem
        self.module = module
        self.exp_dtype = exp_dtype

    def __process_modules__(self, process):
        return FiniteFieldPow(process(self.ff), process(self.module), self.exp_dtype)


def pow(exp_dtype, ff_elem=None):
    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('pow_def'),
        render_kwds=dict(
            ff=ff_elem, ff_elem=ff_elem.module, mul=mul(ff_elem).module, exp_dtype=exp_dtype))
    return FiniteFieldPow(ff_elem, module, exp_dtype)


class FiniteFieldInvPow2:

    def __init__(self, ff_elem, module, exp_dtype):
        self.ff = ff_elem
        self.module = module
        self.exp_dtype = exp_dtype

    def __process_modules__(self, process):
        return FiniteFieldInvPow2(process(self.ff), process(self.module), self.exp_dtype)


def inv_pow2(exp_dtype, ff_elem=None):
    # this can be relaxed when we witch to regular C
    assert numpy.dtype(exp_dtype) == numpy.dtype('uint32')

    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('inv_pow2_def'),
        render_kwds=dict(ff=ff_elem, ff_elem=ff_elem.module, exp_dtype=exp_dtype))
    return FiniteFieldPow(ff_elem, module, exp_dtype)


class FiniteFieldLsh:

    def __init__(self, ff_elem, module, exp_dtype):
        self.ff = ff_elem
        self.module = module
        self.exp_dtype = exp_dtype

    def __process_modules__(self, process):
        return FiniteFieldLsh(process(self.ff), process(self.module), self.exp_dtype)


def lsh(exp_range, exp_dtype, ff_elem=None, method="c", nested_method="c"):
    # this can be relaxed when we witch to regular C
    assert numpy.dtype(exp_dtype) == numpy.dtype('uint32')

    assert exp_range in (32, 64, 96, 128, 160, 192)

    if ff_elem is None:
        ff_elem = get_ff_elem()
    module = Module(
        TEMPLATE.get_def('lsh_def'),
        render_kwds=dict(
            method=method,
            sub=sub(ff_elem=ff_elem, method=nested_method).module, # used in "c" method
            add=add(ff_elem=ff_elem, method=nested_method).module, # used in "c" method
            mod=mod(ff_elem=ff_elem, method=nested_method).module, # used in "c_from_asm" method
            exp_range=exp_range, ff=ff_elem, ff_elem=ff_elem.module,
            exp_dtype=exp_dtype))
    return FiniteFieldLsh(ff_elem, module, exp_dtype)
