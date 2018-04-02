import numpy

from .lwe import *
from .tgsw import *
from .tlwe import *
from .lwe_bootstrapping import *


class TFHEParameters:

    def __init__(self):
        # In the reference implementation there was a parameter `minimum_lambda` here,
        # which was unused.

        # the parameters are only implemented for about 128bit of security!

        mul_by_sqrt_two_over_pi = lambda x: x * (2 / numpy.pi)**0.5

        N = 1024
        k = 1
        n = 500
        bk_l = 2
        bk_Bgbit = 10
        ks_basebit = 2
        ks_length = 8
        ks_stdev = mul_by_sqrt_two_over_pi(1/2**15) # standard deviation
        bk_stdev = mul_by_sqrt_two_over_pi(9e-9) # standard deviation
        max_stdev = mul_by_sqrt_two_over_pi(1/2**4 / 4) # max standard deviation for a 1/4 msg space

        params_in = LweParams(n, ks_stdev, max_stdev)
        params_accum = TLweParams(N, k, bk_stdev, max_stdev)
        params_bk = TGswParams(bk_l, bk_Bgbit, params_accum)

        self.ks_t = ks_length
        self.ks_basebit = ks_basebit
        self.in_out_params = params_in
        self.tgsw_params = params_bk


class TFHESecretKey:

    def __init__(self, params: TFHEParameters, lwe_key: LweKey, tgsw_key: TGswKey):
        self.params = params
        self.lwe_key = lwe_key
        self.tgsw_key = tgsw_key


class TFHECloudKey:

    def __init__(self, params: TFHEParameters, bkFFT: LweBootstrappingKeyFFT):
        self.params = params
        self.bkFFT = bkFFT


def tfhe_parameters(key): # union(TFHESecretKey, TFHECloudKey)
    return key.params


def tfhe_key_pair(rng):
    params = TFHEParameters()

    lwe_key = LweKey.from_rng(rng, params.in_out_params)
    tgsw_key = TGswKey(rng, params.tgsw_params)
    secret_key = TFHESecretKey(params, lwe_key, tgsw_key)

    bkFFT = LweBootstrappingKeyFFT(rng, params.ks_t, params.ks_basebit, lwe_key, tgsw_key)
    cloud_key = TFHECloudKey(params, bkFFT)

    return secret_key, cloud_key


def tfhe_encrypt(rng, key: TFHESecretKey, message):
    result = empty_ciphertext(key.params, message.shape)
    _1s8 = modSwitchToTorus32(1, 8)
    mus = numpy.array([_1s8 if bit else -_1s8 for bit in message])
    alpha = key.params.in_out_params.alpha_min # TODO: specify noise
    lweSymEncrypt(rng, result, mus, alpha, key.lwe_key)
    return result


def tfhe_decrypt(key: TFHESecretKey, ciphertext: LweSampleArray):
    mus = lwePhase(ciphertext, key.lwe_key)
    return numpy.array([(mu > 0) for mu in mus])


def empty_ciphertext(params: TFHEParameters, shape):
    return LweSampleArray(params.in_out_params, shape)
