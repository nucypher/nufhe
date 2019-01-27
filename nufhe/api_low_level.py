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

import io
import pickle

import numpy

from .numeric_functions import phase_to_t32
from .lwe import LweParams, LweKey, LweSampleArray, lwe_encrypt, lwe_decrypt, LweKeyswitchKey
from .tgsw import TGswParams, TGswKey
from .tlwe import TLweParams
from .bootstrap import BootstrapKey
from .performance import PerformanceParameters, PerformanceParametersForDevice


class NuFHEParameters:
    """
    Parameters of the FHE scheme.

    :param transform_type: ``'NTT'`` or ``'FFT'``, specifying the transform to be used for
        internal purposes. ``'FFT'`` is generally faster, but may not be supported on
        some videocards (since it requires double precision floating point numbers).

    .. note::

        The default parameters correspond to about 128 bits of security.
    """

    def __init__(self, transform_type='NTT', tlwe_mask_size=1):

        assert transform_type in ('FFT', 'NTT')
        assert tlwe_mask_size >= 1

        tlwe_polynomial_degree = 1024
        lwe_size = 500

        bs_decomp_length = 2 # bootstrap decomposition length
        bs_log2_base = 10 # bootstrap log2(decomposition_base)

        ks_decomp_length = 8 # keyswitch decomposition length (the precision of the keyswitch)
        ks_log2_base = 2 # keyswitch log2(decomposition base)

        coeff = (2 / numpy.pi)**0.5
        ks_stdev = 1/2**15 * coeff # keyswitch minimal standard deviation
        bs_stdev = 9e-9 * coeff # bootstrap minimal standard deviation
        max_stdev = 1/2**4 / 4 * coeff # max standard deviation for a 1/4 msg space

        params_in = LweParams(lwe_size, ks_stdev, max_stdev)
        params_accum = TLweParams(
            tlwe_polynomial_degree, tlwe_mask_size, bs_stdev, max_stdev, transform_type)
        params_bs = TGswParams(params_accum, bs_decomp_length, bs_log2_base)

        self.ks_decomp_length = ks_decomp_length
        self.ks_log2_base = ks_log2_base
        self.in_out_params = params_in
        self.tgsw_params = params_bs

        self._transform_type = transform_type
        self._tlwe_mask_size = tlwe_mask_size

    def __hash__(self):
        return hash((
            self.__class__,
            self._transform_type,
            self._tlwe_mask_size,
            ))

    def __eq__(self, other: 'NuFHEParameters'):
        return (
            self.__class__ == other.__class__
            and self._transform_type == other._transform_type
            and self._tlwe_mask_size == other._tlwe_mask_size)


class NuFHESecretKey:
    """
    A secret key for the FHE scheme.

        .. py:attribute:: params

            A :py:class:`~nufhe.NuFHEParameters` object.
    """

    def __init__(self, params: NuFHEParameters, lwe_key: LweKey):
        """__init__()""" # hide the signature from Sphinx
        self.params = params
        self.lwe_key = lwe_key

    @classmethod
    def from_rng(cls, thr, params: NuFHEParameters, rng):
        """
        Generate a new secret key.

        :param thr: a ``reikna`` ``Thread`` object.
        :param params: FHE scheme parameters.
        :param rng: an RNG object, one of :ref:`random-number-generators`.
        """
        lwe_key = LweKey.from_rng(thr, params.in_out_params, rng)
        return cls(params, lwe_key)

    def dump(self, file_obj):
        """
        Serialize into the given ``file_obj``, a writeable file-like object.
        """
        pickle.dump(self.params, file_obj)
        self.lwe_key.dump(file_obj)

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
        lwe_key = LweKey.load(file_obj, thr)
        return cls(params, lwe_key)

    @classmethod
    def loads(cls, s: bytes, thr):
        """
        Deserialize from the given bytestring
        using the ``reikna`` thread ``thr`` to store arrays.
        """
        file_obj = io.BytesIO(s)
        return cls.load(file_obj, thr)

    def __eq__(self, other: 'NuFHESecretKey'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and self.lwe_key == other.lwe_key)


class NuFHECloudKey:
    """
    A cloud key for the FHE scheme.

        .. py:attribute:: params

            A :py:class:`~nufhe.NuFHEParameters` object.
    """

    def __init__(
            self, params: NuFHEParameters,
            bootstrap_key: BootstrapKey, keyswitch_key: LweKeyswitchKey):
        """__init__()""" # hide the signature from Sphinx
        self.params = params
        self.bootstrap_key = bootstrap_key
        self.keyswitch_key = keyswitch_key

    @classmethod
    def from_rng(
            cls, thr, params: NuFHEParameters, rng, secret_key: NuFHESecretKey,
            perf_params: PerformanceParametersForDevice=None):
        """
        Generate a new cloud key based on the given secret key.

        :param thr: a ``reikna`` ``Thread`` object.
        :param params: FHE scheme parameters.
        :param rng: an RNG object, one of :ref:`random-number-generators`.
        :param secret_key: the secret key object.
        :param perf_params: an override for performance parameters.
        """

        if perf_params is None:
            perf_params = PerformanceParameters(params).for_device(thr.device_params)

        tgsw_key = TGswKey.from_rng(thr, params.tgsw_params, rng)
        bk = BootstrapKey.from_rng(thr, rng, secret_key.lwe_key, tgsw_key, perf_params)
        ks = LweKeyswitchKey.from_tgsw_key(
            thr, rng, params.ks_decomp_length, params.ks_log2_base,
            secret_key.lwe_key, tgsw_key)
        return cls(params, bk, ks)

    def dump(self, file_obj):
        """
        Serialize into the given ``file_obj``, a writeable file-like object.
        """
        pickle.dump(self.params, file_obj)
        self.bootstrap_key.dump(file_obj)
        self.keyswitch_key.dump(file_obj)

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
        bootstrap_key = BootstrapKey.load(file_obj, thr)
        keyswitch_key = LweKeyswitchKey.load(file_obj, thr)
        return cls(params, bootstrap_key, keyswitch_key)

    @classmethod
    def loads(cls, s: bytes, thr):
        """
        Deserialize from the given bytestring
        using the ``reikna`` thread ``thr`` to store arrays.
        """
        file_obj = io.BytesIO(s)
        return cls.load(file_obj, thr)

    def __eq__(self, other: 'NuFHECloudKey'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and self.bootstrap_key == other.bootstrap_key
            and self.keyswitch_key == other.keyswitch_key)


def make_key_pair(thr, rng, **params):
    """
    Creates a pair of :py:class:`NuFHESecretKey` and :py:class:`NuFHECloudKey`
    corresponding to :py:class:`~nufhe.NuFHEParameters` created with keywords ``params``.
    """
    nufhe_params = NuFHEParameters(**params)
    secret_key = NuFHESecretKey.from_rng(thr, nufhe_params, rng)
    cloud_key = NuFHECloudKey.from_rng(thr, nufhe_params, rng, secret_key)
    return secret_key, cloud_key


_1s8 = phase_to_t32(1, 8)


@numpy.vectorize
def bool_to_t32(bit):
    return _1s8 if bit else -_1s8


@numpy.vectorize
def t32_to_bool(mu):
    return mu > 0


def encrypt(thr, rng, key: NuFHESecretKey, message):
    """
    Encrypts a message.

    :param rng: an RNG object, one of :ref:`random-number-generators`.
    :param key: the secret key.
    :param message: a ``numpy`` array of bit values to encrypt;
        if the ``dtype`` is not ``numpy.bool``, it will be converted to ``numpy.bool``.
    :returns: an :py:class:`LweSampleArray` object with the same `shape` as the given array.
    """
    message = numpy.asarray(message)
    result = empty_ciphertext(thr, key.params, message.shape)
    mus = thr.to_device(bool_to_t32(message))
    noise = key.params.in_out_params.min_noise
    lwe_encrypt(thr, rng, result, mus, noise, key.lwe_key)
    return result


def decrypt(thr, key: NuFHESecretKey, ciphertext: LweSampleArray):
    """
    Decrypts a message.

    :param key: the secret key.
    :param ciphertext: an encrypted message.
    :returns: a ``numpy.ndarray`` object of the type ``numpy.bool``
        and the same `shape` as ``ciphertext``.
    """

    mus = lwe_decrypt(thr, ciphertext, key.lwe_key)
    return t32_to_bool(mus)


def empty_ciphertext(thr, params: NuFHEParameters, shape):
    """
    Creates an uninitialized :py:class:`LweSampleArray` with the shape ``shape``.
    """
    return LweSampleArray.empty(thr, params.in_out_params, shape)
