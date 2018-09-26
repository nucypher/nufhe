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

import pickle

import numpy

from .numeric_functions import phase_to_t32
from .lwe import LweParams, LweKey, LweSampleArray, lwe_encrypt, lwe_decrypt, LweKeyswitchKey
from .tgsw import TGswParams, TGswKey
from .tlwe import TLweParams
from .bootstrap import BootstrapKey
from .performance import PerformanceParameters


class NuFHEParameters:

    def __init__(self, transform_type='NTT', tlwe_mask_size=1):
        # Note: the default parameters correspond to about 128bit of security!

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

    def __init__(self, params: NuFHEParameters, lwe_key: LweKey):
        self.params = params
        self.lwe_key = lwe_key

    @classmethod
    def from_rng(cls, thr, params: NuFHEParameters, rng):
        lwe_key = LweKey.from_rng(thr, params.in_out_params, rng)
        return cls(params, lwe_key)

    def dump(self, file_obj):
        pickle.dump(self.params, file_obj)
        self.lwe_key.dump(file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        params = pickle.load(file_obj)
        lwe_key = LweKey.load(file_obj, thr)
        return cls(params, lwe_key)

    def __eq__(self, other: 'NuFHESecretKey'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and self.lwe_key == other.lwe_key)


class NuFHECloudKey:

    def __init__(
            self, params: NuFHEParameters,
            bootstrap_key: BootstrapKey, keyswitch_key: LweKeyswitchKey):
        self.params = params
        self.bootstrap_key = bootstrap_key
        self.keyswitch_key = keyswitch_key

    @classmethod
    def from_rng(
            cls, thr, params: NuFHEParameters, rng, secret_key: NuFHESecretKey,
            perf_params: PerformanceParameters):
        tgsw_key = TGswKey.from_rng(thr, params.tgsw_params, rng)
        bk = BootstrapKey.from_rng(thr, rng, secret_key.lwe_key, tgsw_key, perf_params)
        ks = LweKeyswitchKey.from_tgsw_key(
            thr, rng, params.ks_decomp_length, params.ks_log2_base,
            secret_key.lwe_key, tgsw_key)
        return cls(params, bk, ks)

    def dump(self, file_obj):
        pickle.dump(self.params, file_obj)
        self.bootstrap_key.dump(file_obj)
        self.keyswitch_key.dump(file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        params = pickle.load(file_obj)
        bootstrap_key = BootstrapKey.load(file_obj, thr)
        keyswitch_key = LweKeyswitchKey.load(file_obj, thr)
        return cls(params, bootstrap_key, keyswitch_key)

    def __eq__(self, other: 'NuFHECloudKey'):
        return (
            self.__class__ == other.__class__
            and self.params == other.params
            and self.bootstrap_key == other.bootstrap_key
            and self.keyswitch_key == other.keyswitch_key)


def nufhe_parameters(key): # union(NuFHESecretKey, NuFHECloudKey)
    return key.params


def make_key_pair(thr, rng, **params):

    nufhe_params = NuFHEParameters(**params)

    # TODO: use PerformanceParameters from the user
    perf_params = PerformanceParameters(nufhe_params)

    secret_key = NuFHESecretKey.from_rng(thr, nufhe_params, rng)
    cloud_key = NuFHECloudKey.from_rng(thr, nufhe_params, rng, secret_key, perf_params)

    return secret_key, cloud_key


_1s8 = phase_to_t32(1, 8)

@numpy.vectorize
def _to_mu(bit):
    return _1s8 if bit else -_1s8

@numpy.vectorize
def _from_mu(mu):
    return mu > 0


def encrypt(thr, rng, key: NuFHESecretKey, message):
    result = empty_ciphertext(thr, key.params, message.shape)
    mus = thr.to_device(_to_mu(message))
    noise = key.params.in_out_params.min_noise
    lwe_encrypt(thr, rng, result, mus, noise, key.lwe_key)
    return result


def decrypt(thr, key: NuFHESecretKey, ciphertext: LweSampleArray):
    mus = lwe_decrypt(thr, ciphertext, key.lwe_key)
    return _from_mu(mus)


def empty_ciphertext(thr, params: NuFHEParameters, shape):
    return LweSampleArray.empty(thr, params.in_out_params, shape)
