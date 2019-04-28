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

import numpy

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
    lwe_noiseless_trivial_constant,
    lwe_negate,
    lwe_copy,
    )
from .api_low_level import NuFHECloudKey, bool_to_t32
from .bootstrap import bootstrap
from .performance import PerformanceParameters, PerformanceParametersForDevice


def get_shape(obj):
    if hasattr(obj, 'shape'):
        return obj.shape
    elif isinstance(obj, list):
        return numpy.asarray(obj).shape
    else:
        raise ValueError("An object of type " + str(type(obj)) + " is not array-like")


def _result_shape_pair(shape1, shape2):
    if len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    else:
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1

    if any((l1 != l2 and l1 > 1 and l2 > 1) for l1, l2 in zip(shape1, shape2)):
        raise ValueError("Incompatible shapes: {s1}, {s2}".format(s1=shape1, s2=shape2))

    return tuple((l1 if l1 > 1 else l2) for l1, l2 in zip(shape1, shape2))


def result_shape(*shapes):
    if len(shapes) == 1:
        return shapes[0]
    elif len(shapes) == 2:
        return _result_shape_pair(*shapes)
    else:
        return _result_shape_pair(shapes[0], result_shape(*shapes[1:]))


def check_shape(result, *args):
    rshape = result_shape(*[arg.shape for arg in args])
    if len(rshape) > len(result.shape) or rshape != result.shape[len(result.shape)-len(rshape):]:
        raise ValueError(
            ("The shape of the result derived from the arguments {derived_shape} "
            "cannot be broadcasted to the shape of the destination {dest_shape}").format(
            derived_shape=rshape, dest_shape=result.shape))


def gate_nand(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped NAND gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params

    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,1/8) - a - b
    NandConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, NandConst)
    lwe_sub_to(thr, temp_result, a)
    lwe_sub_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_or(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped OR gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,1/8) + a + b
    OrConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, OrConst)
    lwe_add_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_and(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped AND gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,-1/8) + a + b
    AndConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, AndConst)
    lwe_add_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_xor(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped XOR gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,1/4) + 2*(a + b)
    XorConst = phase_to_t32(1, 4)
    lwe_noiseless_trivial_constant(thr, temp_result, XorConst)
    lwe_add_mul_to(thr, temp_result, 2, a)
    lwe_add_mul_to(thr, temp_result, 2, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_xnor(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped XNOR gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,-1/4) + 2*(-a-b)
    XnorConst = phase_to_t32(-1, 4)
    lwe_noiseless_trivial_constant(thr, temp_result, XnorConst)
    lwe_sub_mul_to(thr, temp_result, 2, a)
    lwe_sub_mul_to(thr, temp_result, 2, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_not(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray, a: LweSampleArray,
        perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic NOT gate.
    Applied elementwise on an encrypted array of bits.

    Not bootstrapped; ``perf_params`` does not have any effect and is only present
    for the sake of API uniformity.

    The shape of ``a`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the source ciphertext.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 1 LWE sample (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)

    check_shape(result, a)

    in_out_params = cloud_key.params.in_out_params
    lwe_negate(thr, result, a)


def gate_copy(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray, a: LweSampleArray,
        perf_params: PerformanceParametersForDevice=None):
    """
    Copy the contents of the ciphertext ``a`` to ``result``.

    Not bootstrapped; ``perf_params`` does not have any effect and is only present
    for the sake of API uniformity.

    The shape of ``a`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the source ciphertext.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 1 LWE sample (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)

    check_shape(result, a)

    in_out_params = cloud_key.params.in_out_params
    lwe_copy(thr, result, a)


"""
 * Homomorphic Trivial Constant gate (doesn't need to be bootstrapped)
 * Takes a boolean value)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
def gate_constant(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray, vals,
        perf_params: PerformanceParametersForDevice=None):
    """
    Fill each bit of the ciphertext ``result`` with the trivial encryption
    of the plaintext values from ``vals`` (which will be converted to ``bool``).

    ``vals`` should be an array or a list with a shape broadcastable to the shape of ``result``,
    or a scalar value.

    .. note::

        "Trivial encryption" means that the result of this gate does not require
        a secret key for decryption, and cannot be used to implement public key encryption.
        Its intended purpose is to initialize constants in bootstrapped circuits.

    Not bootstrapped; ``perf_params`` does not have any effect and is only present
    for the sake of API uniformity.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
        Must be the same shape as the ``vals`` array.
    :param vals: a ``numpy.bool`` array (or anything castable to it) used to fill the ciphertext.
    :param perf_params: an override for performance parameters.
    """

    vals = numpy.asarray(vals)
    vals = bool_to_t32(vals)

    check_shape(result, vals)

    if vals.ndim == 0:
        lwe_noiseless_trivial_constant(thr, result, vals)
    else:
        lwe_noiseless_trivial(thr, result, thr.to_device(vals))


def gate_nor(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped NOR gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,-1/8) - a - b
    NorConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, NorConst)
    lwe_sub_to(thr, temp_result, a)
    lwe_sub_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_andny(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped ANDNY (`(not a) and b`) gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,-1/8) - a + b
    AndNYConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, AndNYConst)
    lwe_sub_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_andyn(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped ANDYN (`a and (not b)`) gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,-1/8) + a - b
    AndYNConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, AndYNConst)
    lwe_add_to(thr, temp_result, a)
    lwe_sub_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_orny(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped ORNY (`(not a) or b`) gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,1/8) - a + b
    OrNYConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, OrNYConst)
    lwe_sub_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_oryn(
        thr, cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped ORYN (`a or (not b)`) gate.
    Applied elementwise on two encrypted arrays of bits.

    The shapes of ``a`` and ``b`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b)

    in_out_params = cloud_key.params.in_out_params
    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)

    #compute: (0,1/8) + a - b
    OrYNConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, OrYNConst)
    lwe_add_to(thr, temp_result, a)
    lwe_sub_to(thr, temp_result, b)

    #if the phase is positive, the result is 1/8
    #if the phase is positive, else the result is -1/8
    MU = phase_to_t32(1, 8)
    bootstrap(
        thr, result, cloud_key.bootstrap_key, cloud_key.keyswitch_key,
        MU, temp_result, perf_params)


def gate_mux(
        thr,
        cloud_key: NuFHECloudKey, result: LweSampleArray,
        a: LweSampleArray, b: LweSampleArray, c: LweSampleArray,
        perf_params: PerformanceParametersForDevice=None):
    """
    Homomorphic bootstrapped MUX (`b if a else c`, or, equivalently,
    `(a and b) or ((not a) and c)`) gate.
    Applied elementwise on three encrypted arrays of bits.

    The shapes of ``a``, ``b`` and ``c`` should be broadcastable to the shape of ``result``.

    :param thr: a ``reikna`` ``Thread`` object.
    :param cloud_key: the cloud key.
    :param result: an empty ciphertext where the result will be stored.
    :param a: the ciphertext with the first argument.
    :param b: the ciphertext with the second argument.
    :param c: the ciphertext with the third argument.
    :param perf_params: an override for performance parameters.
    """

    # * Takes in input 3 LWE samples (with message space [-1/8,1/8], noise<1/16)
    # * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)

    if perf_params is None:
        perf_params = PerformanceParameters(cloud_key.params).for_device(thr.device_params)

    check_shape(result, a, b, c)

    MU = phase_to_t32(1, 8)
    in_out_params = cloud_key.params.in_out_params
    extracted_params = cloud_key.params.tgsw_params.tlwe_params.extracted_lweparams

    temp_result = LweSampleArray.empty(thr, in_out_params, result.shape)
    temp_result1 = LweSampleArray.empty(thr, extracted_params, result.shape)
    u1 = LweSampleArray.empty(thr, extracted_params, result.shape)
    u2 = LweSampleArray.empty(thr, extracted_params, result.shape)

    #compute "AND(a,b)": (0,-1/8) + a + b
    AndConst = phase_to_t32(-1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result, AndConst)
    lwe_add_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, b)
    # Bootstrap without KeySwitch
    bootstrap(
        thr, u1, cloud_key.bootstrap_key, cloud_key.keyswitch_key, MU, temp_result,
        perf_params, no_keyswitch=True)

    #compute "AND(not(a),c)": (0,-1/8) - a + c
    lwe_noiseless_trivial_constant(thr, temp_result, AndConst)
    lwe_sub_to(thr, temp_result, a)
    lwe_add_to(thr, temp_result, c)
    # Bootstrap without KeySwitch
    bootstrap(
        thr, u2, cloud_key.bootstrap_key, cloud_key.keyswitch_key, MU, temp_result,
        perf_params, no_keyswitch=True)

    # Add u1=u1+u2
    MuxConst = phase_to_t32(1, 8)
    lwe_noiseless_trivial_constant(thr, temp_result1, MuxConst)
    lwe_add_to(thr, temp_result1, u1)
    lwe_add_to(thr, temp_result1, u2)

    # Key switching
    lwe_keyswitch(thr, result, cloud_key.keyswitch_key, temp_result1)
