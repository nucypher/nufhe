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

import time

from .numeric_functions import phase_to_t32
from .lwe import (
    LweSampleArray,
    lwe_keyswitch,
    )
from .lwe import (
    lwe_add_to,
    lwe_sub_to,
    lwe_add_mul_to,
    lwe_sub_mul_to,
    lwe_noiseless_trivial,
    lwe_negate,
    lwe_copy,
    )
from .keys import NuFHECloudKey
from .bootstrap import bootstrap
from .performance import performance_parameters


def result_shape(shape1, shape2):
    if len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    else:
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1

    if any((l1 != l2 and l1 > 1 and l2 > 1) for l1, l2 in zip(shape1, shape2)):
        raise ValueError("Incompatible shapes: {s1}, {s2}".format(s1=shape1, s2=shape2))

    return tuple((l1 if l1 > 1 else l2) for l1, l2 in zip(shape1, shape2))


"""
 * Homomorphic bootstrapped NAND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_nand(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params

    temp_result = LweSampleArray.empty(thr, in_out_params, rshape)

    #compute: (0,1/8) - ca - cb
    NandConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial(thr, temp_result, NandConst)
    lwe_sub_to(thr, temp_result, ca)
    lwe_sub_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped OR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_or(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,1/8) + ca + cb
    OrConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial(thr, temp_result, OrConst)
    lwe_add_to(thr, temp_result, ca)
    lwe_add_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped AND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_and(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,-1/8) + ca + cb
    AndConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial(thr, temp_result, AndConst)
    lwe_add_to(thr, temp_result, ca)
    lwe_add_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped XOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_xor(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,1/4) + 2*(ca + cb)
    XorConst = phase_to_t32(1, 4)
    lwe_noiseless_trivial(thr, temp_result, XorConst)
    lwe_add_mul_to(thr, temp_result, 2, ca)
    lwe_add_mul_to(thr, temp_result, 2, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped XNOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_xnor(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,-1/4) + 2*(-ca-cb)
    XnorConst = phase_to_t32(-1, 4)
    lwe_noiseless_trivial(thr, temp_result, XnorConst)
    lwe_sub_mul_to(thr, temp_result, 2, ca)
    lwe_sub_mul_to(thr, temp_result, 2, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped NOT gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_not(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray,
        perf_params=None):
    in_out_params = ck.params.in_out_params
    lwe_negate(thr, result, ca)


"""
 * Homomorphic bootstrapped COPY gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_copy(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray,
        perf_params=None):
    in_out_params = ck.params.in_out_params
    lwe_copy(thr, result, ca)


"""
 * Homomorphic Trivial Constant gate (doesn't need to be bootstrapped)
 * Takes a boolean value)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_constant(thr, ck: NuFHECloudKey, result: LweSampleArray, val):
    in_out_params = ck.params.in_out_params
    MU = phase_to_t32(1, 8)
    lwe_noiseless_trivial(thr, result, MU if val else -MU)


"""
 * Homomorphic bootstrapped NOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_nor(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,-1/8) - ca - cb
    NorConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial(thr, temp_result, NorConst)
    lwe_sub_to(thr, temp_result, ca)
    lwe_sub_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped AndNY Gate: not(a) and b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_andny(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,-1/8) - ca + cb
    AndNYConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial(thr, temp_result, AndNYConst)
    lwe_sub_to(thr, temp_result, ca)
    lwe_add_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped AndYN Gate: a and not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_andyn(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,-1/8) + ca - cb
    AndYNConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial(thr, temp_result, AndYNConst)
    lwe_add_to(thr, temp_result, ca)
    lwe_sub_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped OrNY Gate: not(a) or b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_orny(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,1/8) - ca + cb
    OrNYConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial(thr, temp_result, OrNYConst)
    lwe_sub_to(thr, temp_result, ca)
    lwe_add_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped OrYN Gate: a or not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_oryn(
        thr, ck: NuFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(ca.shape_info.shape, cb.shape_info.shape)
    assert rshape == result.shape_info.shape

    in_out_params = ck.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)

    #compute: (0,1/8) + ca - cb
    OrYNConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial(thr, temp_result, OrYNConst)
    lwe_add_to(thr, temp_result, ca)
    lwe_sub_to(thr, temp_result, cb)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(thr, result, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result, perf_params)


"""
 * Homomorphic bootstrapped Mux(a,b,c) = a?b:c = a*b + not(a)*c
 * Takes in input 3 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_mux(
        thr,
        ck: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, c: LweSampleArray,
        perf_params=None):

    if perf_params is None:
        perf_params = performance_parameters(nufhe_params=ck.params)

    rshape = result_shape(a.shape_info.shape, result_shape(b.shape_info.shape, c.shape_info.shape))
    assert rshape == result.shape_info.shape

    MU = phase_to_t32(1, 8)
    in_out_params = ck.params.in_out_params
    extracted_params = ck.params.tgsw_params.tlwe_params.extracted_lweparams

    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape_info.shape)
    temp_result1 = LweSampleArray.empty(thr, extracted_params, result.shape_info.shape)
    u1 = LweSampleArray.empty(thr, extracted_params, result.shape_info.shape)
    u2 = LweSampleArray.empty(thr, extracted_params, result.shape_info.shape)

    #compute "AND(a,b)": (0,-1/8) + a + b
    AndConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial(thr, temp_result, AndConst)
    lwe_add_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, b)
    # Bootstrap without KeySwitch
    bootstrap(
        thr, u1, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result,
        perf_params, no_keyswitch=True)

    #compute "AND(not(a),c)": (0,-1/8) - a + c
    lwe_noiseless_trivial(thr, temp_result, AndConst)
    lwe_sub_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, c)
    # Bootstrap without KeySwitch
    bootstrap(
        thr, u2, ck.bootstrap_key, ck.keyswitch_key, MU, temp_result,
        perf_params, no_keyswitch=True)

    # Add u1=u1+u2
    MuxConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial(thr, temp_result1, MuxConst)
    lwe_add_to(thr, temp_result1, u1)
    lwe_add_to(thr, temp_result1, u2)

    # Key switching
    lwe_keyswitch(thr, result, ck.keyswitch_key, temp_result1)
