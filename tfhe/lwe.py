"""
LWE (Learning With Errors) functions.
"""

from reikna.cluda.api import Thread
from reikna.core import Type

from .numeric_functions import (
    Torus32,
    Float,
    )
from .lwe_gpu import (
    LweKeyswitch,
    MakeLweKeyswitchKey,
    LweEncrypt,
    LweDecrypt,
    LweLinear,
    LweNoiselessTrivial,
    )
from .random_numbers import (
    rand_uniform_int32,
    rand_gaussian_float,
    rand_uniform_torus32,
    rand_gaussian_torus32,
    )
from .computation_cache import get_computation


class LweParams:

    def __init__(self, size: int, min_noise: float, max_noise: float):
        self.size = size
        self.min_noise = min_noise # the smallest noise that makes it secure
        self.max_noise = max_noise # the biggest noise that allows decryption


class LweKey:

    def __init__(self, params: LweParams, key):
        self.params = params
        self.key = key

    @classmethod
    def from_rng(cls, thr: Thread, params: LweParams, rng):
        return cls(params, rand_uniform_int32(thr, rng, (params.size,)))

    # extractions ring Lwe * Lwe
    @classmethod
    def from_tlwe_key(cls, params: LweParams, tlwe_key: 'TLweKey'):
        poly_degree = tlwe_key.params.polynomial_degree
        mask_size = tlwe_key.params.mask_size
        assert params.size == poly_degree * mask_size

        key = tlwe_key.key.coefs.ravel()

        return cls(params, key)


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
            and self.current_variances == other.current_variances
            )

    def __hash__(self):
        return hash((self.__class__, self.a, self.b, self.current_variances))


class LweSampleArray:

    def __init__(self, params: LweParams, a, b, current_variances):
        self.params = params
        self.a = a
        self.b = b
        self.current_variances = current_variances
        self.shape_info = LweSampleArrayShapeInfo(a, b, current_variances)

    @classmethod
    def empty(cls, thr: Thread, params: LweParams, shape):
        a = thr.array(shape + (params.size,), Torus32)
        b = thr.array(shape, Torus32)
        current_variances = thr.array(shape, Float)
        return cls(params, a, b, current_variances)

    def __getitem__(self, index):
        a_view = self.a[index]
        b_view = self.b[index]
        cv_view = self.current_variances[index]
        return LweSampleArray(self.params, a_view, b_view, cv_view)


class LweKeyswitchKey:

    def __init__(
            self, thr: Thread, rng,
            in_key: LweKey, out_key: LweKey, decomp_length: int, log2_base: int):

        input_size = in_key.params.size
        output_size = out_key.params.size
        noise = out_key.params.min_noise
        base = 2**log2_base

        lwe = LweSampleArray.empty(thr, out_key.params, (input_size, decomp_length, base))

        noises_b = rand_gaussian_float(
            thr, rng, noise, (input_size, decomp_length, base - 1))
        noises_a = rand_uniform_torus32(
            thr, rng, (input_size, decomp_length, base - 1, output_size))

        comp = get_computation(
            thr, MakeLweKeyswitchKey,
            input_size, output_size, decomp_length, log2_base, noise)
        comp(lwe.a, lwe.b, lwe.current_variances, in_key.key, out_key.key, noises_a, noises_b)

        self.lwe = lwe
        self.input_size = input_size # length of the input key: s'
        self.output_size = output_size # params of the output key s
        self.decomp_length = decomp_length # decomposition length
        self.log2_base = log2_base # log_2(decomposition base)


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


def lwe_noiseless_trivial(thr: Thread, result: LweSampleArray, mu: Torus32):
    """
    Initialize an LWE sample with (0, mu).
    """
    comp = get_computation(thr, LweNoiselessTrivial, result.shape_info)
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
