import numpy

from .numeric_functions import modSwitchToTorus32
from .lwe import LweParams, LweKey, LweSampleArray, lweSymEncrypt_gpu, lwePhase_gpu
from .tgsw import TGswParams, TGswKey
from .tlwe import TLweParams
from .lwe_bootstrapping import LweBootstrappingKeyFFT
from .performance import performance_parameters


class TFHEParameters:

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
        params_accum = TLweParams(tlwe_polynomial_degree, tlwe_mask_size, bs_stdev, max_stdev, transform_type)
        params_bs = TGswParams(bs_decomp_length, bs_log2_base, params_accum)

        self.ks_decomp_length = ks_decomp_length
        self.ks_log2_base = ks_log2_base
        self.in_out_params = params_in
        self.tgsw_params = params_bs


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


def tfhe_key_pair(thr, rng, **params):
    params = TFHEParameters(**params)

    lwe_key = LweKey.from_rng(thr, params.in_out_params, rng)
    tgsw_key = TGswKey(thr, rng, params.tgsw_params)
    secret_key = TFHESecretKey(params, lwe_key, tgsw_key)

    # TODO: use PerformanceParameters from the user
    perf_params = performance_parameters(tfhe_params=params)

    bkFFT = LweBootstrappingKeyFFT(
        thr, rng, params.ks_decomp_length, params.ks_log2_base, lwe_key, tgsw_key, perf_params)
    cloud_key = TFHECloudKey(params, bkFFT)

    return secret_key, cloud_key


_1s8 = modSwitchToTorus32(1, 8)

@numpy.vectorize
def _to_mu(bit):
    return _1s8 if bit else -_1s8

@numpy.vectorize
def _from_mu(mu):
    return mu > 0


def tfhe_encrypt(thr, rng, key: TFHESecretKey, message):
    result = empty_ciphertext(thr, key.params, message.shape)
    mus = thr.to_device(_to_mu(message))
    alpha = key.params.in_out_params.alpha_min # TODO: specify noise
    lweSymEncrypt_gpu(thr, rng, result, mus, alpha, key.lwe_key)
    return result


def tfhe_decrypt(thr, key: TFHESecretKey, ciphertext: LweSampleArray):
    mus = lwePhase_gpu(thr, ciphertext, key.lwe_key)
    return _from_mu(mus)


def empty_ciphertext(thr, params: TFHEParameters, shape):
    return LweSampleArray.empty(thr, params.in_out_params, shape)
