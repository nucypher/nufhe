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

from .numeric_functions import Torus32, double_to_t32


def vec_mul_mat(a, b):
    return (a * b).sum(-1, dtype=Torus32)


def lwe_encrypt_with_external_noise(
        ks_a, ks_b, ks_cv, messages, noises_a, noises_b, noise: float, key):

    # term h=0 as trivial encryption of 0
    ks_a[:,:,0,:] = 0
    ks_b[:,:,0] = 0
    ks_cv[:,:,0] = 0

    ks_a[:,:,1:,:] = noises_a
    ks_b[:,:,1:] = messages + noises_b + vec_mul_mat(noises_a, key)
    ks_cv[:,:,1:] = noise**2


def MakeLweKeyswitchKeyReference(
        input_size: int, output_size: int, decomp_length: int, log2_base: int, noise: float):

    base = 2**log2_base

    def _kernel(ks_a, ks_b, ks_cv, in_key, out_key, noises_a, noises_b):

        hs = numpy.arange(1, base).astype(Torus32)
        js = numpy.arange(decomp_length).astype(Torus32)

        r_key = in_key[:, None, None]
        r_hs = hs[None, None, :]
        r_js = js[None, :, None]

        messages = r_key * r_hs * (2**(32 - (r_js + 1) * log2_base))

        lwe_encrypt_with_external_noise(
            ks_a, ks_b, ks_cv, messages, noises_a, noises_b, noise, out_key)

    return _kernel


def LweKeyswitchReference(
        shape_info, input_size: int, output_size: int, decomp_length: int, log2_base: int):

    def _kernel(result_a, result_b, result_cv, ks_a, ks_b, ks_cv, source_a, source_b):

        base = 2**log2_base
        prec_offset = 2**(32 - (1 + log2_base * decomp_length)) # precision
        mask = base - 1

        js = numpy.arange(1, decomp_length + 1).reshape(
            (1,) * len(source_a.shape) + (decomp_length,))
        source_a = source_a.reshape(source_a.shape + (1,))
        aijs = (((source_a + prec_offset) >> (32 - js * log2_base)) & mask)

        # Starting from a noiseless trivial LWE:
        # a = 0, b = bi, current_variances = 0
        result_a.fill(0)
        numpy.copyto(result_b, source_b)
        result_cv.fill(0)

        for l in range(input_size):
            for j in range(decomp_length):
                x = aijs.take(l, axis=-2).take(j, axis=-1)
                lwe_sub_to(result_a, result_b, result_cv, ks_a[l,j,x], ks_b[l,j,x], ks_cv[l,j,x])

    return _kernel


def lwe_sub_to(result_a, result_b, result_cv, source_a, source_b, source_cv):
    result_a -= source_a
    result_b -= source_b
    result_cv += source_cv


def LweEncryptReference(shape, lwe_size: int, noise: float):

    def _kernel(result_a, result_b, result_cv, messages, key, noises_a, noises_b):
        numpy.copyto(result_b, noises_b + messages)
        numpy.copyto(result_a, noises_a)
        result_b += vec_mul_mat(result_a, key)
        result_cv.fill(noise**2)

    return _kernel


def LweDecryptReference(shape, lwe_size: int):

    def _kernel(result, lwe_a, lwe_b, key):
        numpy.copyto(result, lwe_b - vec_mul_mat(lwe_a, key))

    return _kernel


def LweLinearReference(result_shape_info, source_shape_info, add_result=False):

    def _kernel(result_a, result_b, result_cv, source_a, source_b, source_cv, p):
        p = Torus32(p)
        numpy.copyto(result_a, (result_a if add_result else 0) + p * source_a)
        numpy.copyto(result_b, (result_b if add_result else 0) + p * source_b)
        numpy.copyto(result_cv, (result_cv if add_result else 0) + p**2 * source_cv)

    return _kernel


def LweNoiselessTrivialReference(result_shape_info, source_shape):

    def _kernel(result_a, result_b, result_cv, mus):
        result_a.fill(0)
        numpy.copyto(result_b, mus)
        result_cv.fill(0)

    return _kernel


def LweNoiselessTrivialConstantReference(result_shape_info):

    def _kernel(result_a, result_b, result_cv, mu):
        result_a.fill(0)
        result_b.fill(mu)
        result_cv.fill(0)

    return _kernel
