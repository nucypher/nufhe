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

from .numeric_functions import phase_to_t32
from .lwe import LweParams, LweKey, LweSampleArray, lwe_encrypt, lwe_decrypt, LweKeyswitchKey
from .tgsw import TGswParams, TGswKey
from .tlwe import TLweParams
from .bootstrap import BootstrapKey
from .performance import performance_parameters


class NuFHEParameters:

    def __init__(self, transform_type='FFT', tlwe_mask_size=1):
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


class NuFHESecretKey:

    def __init__(self, params: NuFHEParameters, lwe_key: LweKey, tgsw_key: TGswKey):
        self.params = params
        self.lwe_key = lwe_key
        self.tgsw_key = tgsw_key


class NuFHECloudKey:

    def __init__(
            self, params: NuFHEParameters,
            bootstrap_key: BootstrapKey, keyswitch_key: LweKeyswitchKey):
        self.params = params
        self.bootstrap_key = bootstrap_key
        self.keyswitch_key = keyswitch_key


def nufhe_parameters(key): # union(NuFHESecretKey, NuFHECloudKey)
    return key.params


def make_key_pair(thr, rng, **params):
    params = NuFHEParameters(**params)

    lwe_key = LweKey.from_rng(thr, params.in_out_params, rng)
    tgsw_key = TGswKey(thr, params.tgsw_params, rng)
    secret_key = NuFHESecretKey(params, lwe_key, tgsw_key)

    # TODO: use PerformanceParameters from the user
    perf_params = performance_parameters(nufhe_params=params)

    bk = BootstrapKey(thr, rng, lwe_key, tgsw_key, perf_params)
    ks = LweKeyswitchKey.from_tgsw_key(
        thr, rng, params.ks_decomp_length, params.ks_log2_base, lwe_key, tgsw_key)
    cloud_key = NuFHECloudKey(params, bk, ks)

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
