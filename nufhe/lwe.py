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

"""
LWE (Learning With Errors) functions.
"""

import io
import pickle

import numpy

from reikna.cluda.api import Thread
from reikna.core import Type
import reikna

from .utils import arrays_equal
from .numeric_functions import (
    Torus32,
    ErrorFloat,
    )
from .lwe_gpu import (
    LweKeyswitch,
    MakeLweKeyswitchKey,
    LweEncrypt,
    LweDecrypt,
    LweLinear,
    LweNoiselessTrivial,
    LweNoiselessTrivialConstant,
    )
from .random_numbers import (
    rand_uniform_bool,
    rand_uniform_torus32,
    rand_gaussian_torus32,
    )
from .computation_cache import get_computation


class LweParams:

    def __init__(self, size: int, min_noise: float, max_noise: float):
        self.size = size
        self.min_noise = min_noise # the smallest noise that makes it secure
        self.max_noise = max_noise # the biggest noise that allows decryption

    def __eq__(self, other: 'LweParams'):
        return (
            self.__class__ == other.__class__
            and self.size == other.size
            and self.min_noise == other.min_noise
            and self.max_noise == other.max_noise)

    def __hash__(self):
        return hash((self.__class__, self.size, self.min_noise, self.max_noise))


class LweKey:

    def __init__(self, params: LweParams, key):
        self.params = params
        self.key = key

    @classmethod
    def from_rng(cls, thr: Thread, params: LweParams, rng):
        return cls(params, rand_uniform_bool(thr, rng, (params.size,)))

    # extractions ring Lwe * Lwe
    @classmethod
    def from_tlwe_key(cls, params: LweParams, tlwe_key: 'TLweKey'):
        poly_degree = tlwe_key.params.polynomial_degree
        mask_size = tlwe_key.params.mask_size
        assert params.size == poly_degree * mask_size

        key = tlwe_key.key.coeffs.ravel()

        return cls(params, key)

    def dump(self, file_obj):
        pickle.dump(self.params, file_obj)
        pickle.dump(self.key.get(), file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        params = pickle.load(file_obj)
        key = pickle.load(file_obj)
        return cls(params, thr.to_device(key))

    def __eq__(self, other: 'LweKey'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and arrays_equal(self.key, other.key))


class LweSampleArrayShapeInfo:

    def __init__(self, a, b, current_variances):

        if (not (len(a.shape) - 1 == len(b.shape) == len(current_variances.shape))
                or not (a.shape[:-1] == b.shape == current_variances.shape)):

            raise ValueError("Inconsistent shapes: {a}, {b}, {cv}".format(
                a=a.shape, b=b.shape, cv=current_variances.shape))

        self.a = Type.from_value(a)
        self.b = Type.from_value(b)
        self.current_variances = Type.from_value(current_variances)
        self.shape = b.shape

    def __eq__(self, other: 'LweSampleArrayShapeInfo'):
        return (
            self.__class__ == other.__class__
            and self.a == other.a
            and self.b == other.b
            and self.current_variances == other.current_variances)

    def __hash__(self):
        return hash((self.__class__, self.a, self.b, self.current_variances))


class LweSampleArray:
    """
    A ciphertext object.

    .. py:attribute:: shape

        The shape of the encrypted plaintext message.
    """

    def __init__(self, params: LweParams, a, b, current_variances):
        """__init__()""" # hide the signature from Sphinx
        self.params = params
        self.a = a
        self.b = b
        self.current_variances = current_variances
        self.shape_info = LweSampleArrayShapeInfo(a, b, current_variances)

    @classmethod
    def empty(cls, thr: Thread, params: LweParams, shape):
        a = thr.array(shape + (params.size,), Torus32)
        b = thr.array(shape, Torus32)
        current_variances = thr.array(shape, ErrorFloat)
        return cls(params, a, b, current_variances)

    @property
    def shape(self):
        return self.shape_info.shape

    def __getitem__(self, index):
        """
        Returns a view over the ciphertext (still a :py:class:`LweSampleArray` object).
        The indexing works in the same way as if it was a regular ``numpy`` array
        with the shape ``shape``.
        """
        a_view = self.a[index]
        b_view = self.b[index]
        cv_view = self.current_variances[index]
        return LweSampleArray(self.params, a_view, b_view, cv_view)

    def __setitem__(self, index, value):
        if not isinstance(value, LweSampleArray):
            raise ValueError("Only assignment of ciphertexts is supported")
        self.a[index] = value.a
        self.b[index] = value.b
        self.current_variances[index] = value.current_variances

    def copy(self):
        """
        Returns a copy of the ciphertext.
        """
        return LweSampleArray(
            self.params, self.a.copy(), self.b.copy(), self.current_variances.copy())

    def roll(self, shift, axis=-1):
        """
        Cyclically shifts encrypted bits of the cyphertext **inplace**
        by ``shift`` positions to the right along ``axis``.
        ``shift`` can be negative (in which case the elements are shifted to the left).
        Elements that are shifted beyond the last position are re-introduced at the first
        (and vice versa).

        Works equivalently to ``numpy.roll`` (except ``axis=None`` is not supported).
        """
        if shift == 0:
            return

        axis = axis % len(self.shape)

        self.a.roll(shift, axis=axis)
        self.b.roll(shift, axis=axis)
        self.current_variances.roll(shift, axis=axis)

    def dump(self, file_obj):
        """
        Serialize into the given ``file_obj``, a writeable file-like object.
        """
        pickle.dump(self.params, file_obj)
        pickle.dump(self.a.get(), file_obj)
        pickle.dump(self.b.get(), file_obj)
        pickle.dump(self.current_variances.get(), file_obj)

    def dumps(self):
        """
        Serialize into a bytestring.
        """
        file_obj = io.BytesIO()
        self.dump(file_obj)
        return file_obj.getvalue()

    @classmethod
    def load(cls, file_obj, thr):
        """
        Deserialize from the given ``file_obj``, a readable file-like object,
        using the ``reikna`` thread ``thr`` to store arrays.
        """
        params = pickle.load(file_obj)
        a = thr.to_device(pickle.load(file_obj))
        b = thr.to_device(pickle.load(file_obj))
        current_variances = thr.to_device(pickle.load(file_obj))
        return cls(params, a, b, current_variances)

    @classmethod
    def loads(cls, s, thr):
        """
        Deserialize from the given bytestring
        using the ``reikna`` thread ``thr`` to store arrays.
        """
        file_obj = io.BytesIO(s)
        return cls.load(file_obj, thr)

    def __eq__(self, other: 'LweSampleArray'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and arrays_equal(self.a, other.a)
            and arrays_equal(self.b, other.b)
            and arrays_equal(self.current_variances, other.current_variances))


class LweKeyswitchKey:

    def __init__(self, lwe: LweSampleArray):
        input_size, decomp_length, base = lwe.shape

        self.lwe = lwe
        self.input_size = input_size # length of the input key: s'
        self.output_size = lwe.params.size # params of the output key s
        self.decomp_length = decomp_length # decomposition length
        self.log2_base = int(numpy.log2(base)) # log_2(decomposition base)

    @classmethod
    def from_tgsw_key(
            cls, thr, rng, ks_decomp_length: int, ks_log2_base: int,
            lwe_key: LweKey, tgsw_key: 'TGswKey'):

        bk_params = tgsw_key.params
        accum_params = bk_params.tlwe_params
        extract_params = accum_params.extracted_lweparams
        extracted_key = LweKey.from_tlwe_key(extract_params, tgsw_key.tlwe_key)

        in_key = extracted_key
        out_key = lwe_key

        input_size = in_key.params.size
        output_size = out_key.params.size
        noise = out_key.params.min_noise
        base = 2**ks_log2_base

        lwe = LweSampleArray.empty(thr, out_key.params, (input_size, ks_decomp_length, base))

        noises_b = rand_gaussian_torus32(
            thr, rng, 0, noise, (input_size, ks_decomp_length, base - 1), centered=True)
        noises_a = rand_uniform_torus32(
            thr, rng, (input_size, ks_decomp_length, base - 1, output_size))

        comp = get_computation(
            thr, MakeLweKeyswitchKey,
            input_size, output_size, ks_decomp_length, ks_log2_base, noise)
        comp(lwe.a, lwe.b, lwe.current_variances, in_key.key, out_key.key, noises_a, noises_b)

        return cls(lwe)

    def dump(self, file_obj):
        self.lwe.dump(file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        lwe = LweSampleArray.load(file_obj, thr)
        return cls(lwe)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.lwe == other.lwe)


def lwe_keyswitch(thr: Thread, result: LweSampleArray, ks: LweKeyswitchKey, sample: LweSampleArray):
    """
    Translate the message of the result sample by -sum(a[i].s[i]) where s is the secret.
    """
    lwe = ks.lwe
    comp = get_computation(
        thr, LweKeyswitch,
        result.shape_info, ks.input_size, ks.output_size, ks.decomp_length, ks.log2_base)
    # Note: sample.current_variances are ignored.
    comp(
        result.a, result.b, result.current_variances,
        lwe.a, lwe.b, lwe.current_variances, sample.a, sample.b)


def lwe_encrypt(thr: Thread, rng, result: LweSampleArray, messages, noise: float, key: LweKey):
    """
    Encrypt a message with the secret key, with stdev `noise`.
    """
    lwe_size = key.params.size
    noises_b = rand_gaussian_torus32(thr, rng, 0, noise, messages.shape)
    noises_a = rand_uniform_torus32(thr, rng, messages.shape + (lwe_size,))
    comp = get_computation(thr, LweEncrypt, messages.shape, lwe_size, noise)
    comp(result.a, result.b, result.current_variances, messages, key.key, noises_a, noises_b)


def lwe_decrypt(thr: Thread, sample: LweSampleArray, key: LweKey):
    """
    Decrypt an LWE array with the secret key.
    """
    result = thr.empty_like(sample.b)
    comp = get_computation(thr, LweDecrypt, sample.shape_info.shape, key.params.size)
    comp(result, sample.a, sample.b, key.key)
    return result.get()


def lwe_noiseless_trivial(thr: Thread, result: LweSampleArray, mus):
    """
    Initialize LWE samples with `(0, mu)` for each `mu` in `mus`.
    """
    comp = get_computation(thr, LweNoiselessTrivial, result.shape_info, mus.shape)
    comp(result.a, result.b, result.current_variances, mus)


def lwe_noiseless_trivial_constant(thr: Thread, result: LweSampleArray, mu):
    """
    Initialize LWE samples with `(0, mu)`.
    """
    comp = get_computation(thr, LweNoiselessTrivialConstant, result.shape_info)
    comp(result.a, result.b, result.current_variances, mu)


# Arithmetic operations on LWE samples


def lwe_negate(thr: Thread, result: LweSampleArray, source: LweSampleArray):
    """
    result = -sample
    """
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, -1)


def lwe_copy(thr: Thread, result: LweSampleArray, source: LweSampleArray):
    """
    result = sample
    """
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, 1)


def lwe_add_to(thr: Thread, result: LweSampleArray, source: LweSampleArray):
    """
    result += sample
    """
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, 1)


def lwe_add_mul_to(thr: Thread, result: LweSampleArray, p: int, source: LweSampleArray):
    """
    result += p * sample
    """
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, p)


def lwe_sub_to(thr: Thread, result: LweSampleArray, source: LweSampleArray):
    """
    result -= sample
    """
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, -1)


def lwe_sub_mul_to(thr: Thread, result: LweSampleArray, p: int, source: LweSampleArray):
    """
    result -= p * sample
    """
    comp = get_computation(thr, LweLinear, result.shape_info, source.shape_info, add_result=True)
    comp(
        result.a, result.b, result.current_variances,
        source.a, source.b, source.current_variances, -p)


def concatenate(lwe_sample_arrays, axis=0, out=None):
    """
    Concatenates several ciphertext arrays along ``axis``.
    """
    if len(lwe_sample_arrays) == 0:
        raise ValueError("Need at least one ciphertext to concatenate")

    params = lwe_sample_arrays[0].params
    lwes_a = [lwe.a for lwe in lwe_sample_arrays]
    lwes_b = [lwe.b for lwe in lwe_sample_arrays]
    lwes_cv = [lwe.current_variances for lwe in lwe_sample_arrays]
    if out is None:
        out = LweSampleArray(
            params,
            reikna.concatenate(lwes_a, axis=axis),
            reikna.concatenate(lwes_b, axis=axis),
            reikna.concatenate(lwes_cv, axis=axis))
    else:
        reikna.concatenate(lwes_a, axis=axis, out=out.a)
        reikna.concatenate(lwes_b, axis=axis, out=out.b)
        reikna.concatenate(lwes_cv, axis=axis, out=out.current_variances)

    return out
