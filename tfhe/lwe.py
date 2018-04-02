import numpy

from .numeric_functions import *


class LweParams:

    def __init__(self, n: int, alpha_min: float, alpha_max: float):
        self.n = n
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max


class LweKey:

    def __init__(self, params: LweParams, key):
        self.params = params
        self.key = key # 1D array of Int32

    @classmethod
    def from_rng(cls, rng, params: LweParams):
        return cls(params, rand_uniform_int32(rng, (params.n,)))

    # extractions Ring Lwe . Lwe
    @classmethod
    def from_key(cls, params: LweParams, tlwe_key):  # sans doute un param supplémentaire
        # TYPING: tlwe_key: TLweKey
        N = tlwe_key.params.N
        k = tlwe_key.params.k
        assert params.n == k * N

        # GPU: array operation
        key = tlwe_key.key.coefs.flatten() # TODO: use an approprtiate method

        return cls(params, key)


class LweSampleArray:

    def __init__(self, params: LweParams, shape):
        self.a = numpy.empty(shape + (params.n,), Torus32)
        self.b = numpy.empty(shape, Torus32)
        self.current_variances = numpy.empty(shape, numpy.float64)
        self.shape = shape
        self.params = params

    def __getitem__(self, *args):
        sub_a = self.a[args]
        sub_b = self.b[args]
        sub_cv = self.current_variances[args]
        res = LweSampleArray(self.params, sub_b.shape)

        res.a = sub_a
        res.b = sub_b
        res.current_variances = sub_cv

        return res


def vec_mul_mat(b, a):
    return (a * b).sum(-1, dtype=numpy.int32)


# * This function encrypts message by using key, with stdev alpha
# * The Lwe sample for the result must be allocated and initialized
# * (this means that the parameters are already in the result)
def lweSymEncrypt(rng, result: LweSampleArray, messages, alpha: float, key: LweKey):
    # TYPING: messages: Array{Torus32}

    assert result.shape == messages.shape

    n = key.params.n

    result.b = rand_gaussian_torus32(rng, 0, alpha, messages.shape) + messages
    result.a = rand_uniform_torus32(rng, messages.shape + (n,))
    result.b += vec_mul_mat(key.key, result.a)
    result.current_variances.fill(alpha**2)


# This function computes the phase of sample by using key : phi = b - a.s
def lwePhase(sample: LweSampleArray, key: LweKey):
    return sample.b - vec_mul_mat(key.key, sample.a)


# Arithmetic operations on Lwe samples


# result = sample
def lweCopy(result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    result.a = sample.a.copy()
    result.b = sample.b.copy()
    result.current_variances = sample.current_variances.copy()


# result = -sample
def lweNegate(result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    result.a = -sample.a
    result.b = -sample.b
    result.current_variances = sample.current_variances.copy()


# result = (0,mu)
def lweNoiselessTrivial(result: LweSampleArray, mus, params: LweParams):
    # TYPING: mus: Union{Array{Torus32}, Torus32}
    # GPU: array operations
    result.a.fill(0)
    numpy.copyto(result.b, mus)
    result.current_variances.fill(0)


# result = result + sample
def lweAddTo(result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    # GPU: array operations or a custom kernel
    result.a += sample.a
    result.b += sample.b
    result.current_variances += sample.current_variances


# result = result - sample
def lweSubTo(result: LweSampleArray, sample: LweSampleArray, params: LweParams):
    result.a -= sample.a
    result.b -= sample.b
    result.current_variances += sample.current_variances


# result = result + p.sample
def lweAddMulTo(result: LweSampleArray, p: numpy.int32, sample: LweSampleArray, params: LweParams):
    result.a += p * sample.a
    result.b += p * sample.b
    result.current_variances += p**2 * sample.current_variances


# result = result - p.sample
def lweSubMulTo(result: LweSampleArray, p: numpy.int32, sample: LweSampleArray, params: LweParams):
    result.a -= p * sample.a
    result.b -= p * sample.b
    result.current_variances += p**2 * sample.current_variances


# This function encrypts a message by using key and a given noise value
def lweSymEncryptWithExternalNoise(
        rng, result: LweSampleArray, messages, noises, alpha: float, key: LweKey):

    # TYPING: messages: Array{Torus32}
    # TYPING: noises: Array{Float64}

    #@assert size(result) == size(messages)
    #@assert size(result) == size(noises)

    # GPU: will be made into a kernel

    # term h=0 as trivial encryption of 0 (it will not be used in the KeySwitching)
    result.a[:,:,0,:] = 0
    result.b[:,:,0] = 0
    result.current_variances[:,:,0] = 0

    n = key.params.n

    result.b[:,:,1:] = messages + dtot32(noises)
    result.a[:,:,1:,:] = rand_uniform_torus32(rng, messages.shape + (n,))
    result.b[:,:,1:] += vec_mul_mat(key.key, result.a[:,:,1:,:])
    result.current_variances[:,:,1:] = alpha**2


class LweKeySwitchKey:

    """
    Create the key switching key:
     * normalize the error in the beginning
     * chose a random vector of gaussian noises (same size as ks)
     * recenter the noises
     * generate the ks by creating noiseless encryprions and then add the noise
    """
    def __init__(self, rng, n: int, t: int, basebit: int, in_key: LweKey, out_key: LweKey):

        # GPU: will be possibly made into a kernel including lweSymEncryptWithExternalNoise()

        out_params = out_key.params

        base = 1 << basebit
        ks = LweSampleArray(out_params, (n, t, base))

        alpha = out_key.params.alpha_min

        # chose a random vector of gaussian noises
        noises = rand_gaussian_float(rng, alpha, (n, t, base - 1))

        # recenter the noises
        noises -= noises.mean()

        # generate the ks

        # mess::Torus32 = (in_key.key[i] * Int32(h - 1)) * Int32(1 << (32 - j * basebit))
        hs = numpy.arange(2, base+1)
        js = numpy.arange(1, t+1)

        r_key = in_key.key.reshape(n, 1, 1)
        r_hs = hs.reshape(1, 1, base - 1)
        r_js = js.reshape(1, t, 1)

        messages = r_key * (r_hs - 1) * (1 << (32 - r_js * basebit))
        messages = messages.astype(Torus32)

        lweSymEncryptWithExternalNoise(rng, ks, messages, noises, alpha, out_key)

        self.n = n # length of the input key: s'
        self.t = t # decomposition length
        self.basebit = basebit # log_2(base)
        self.base = base # decomposition base: a power of 2
        self.out_params = out_params # params of the output key s
        self.ks = ks # the keyswitch elements: a n.l.base matrix
        # de taille n pointe vers ks1 un tableau dont les cases sont espaceés de ell positions


"""
 * translates the message of the result sample by -sum(a[i].s[i]) where s is the secret
 * embedded in ks.
 * @param result the LWE sample to translate by -sum(ai.si).
 * @param ks The (n x t x base) key switching key
 *        ks[i][j][k] encodes k.s[i]/base^(j+1)
 * @param params The common LWE parameters of ks and result
 * @param ai The input torus array
 * @param n The size of the input key
 * @param t The precision of the keyswitch (technically, 1/2.base^t)
 * @param basebit Log_2 of base
"""
def lweKeySwitchTranslate_fromArray(
        result: LweSampleArray, ks: LweSampleArray, params: LweParams,
        ai, n: int, t: int, basebit: int):

    # TYPING: ai: Array{Torus32, 2}
    # GPU: array operations or (most probably) a custom kernel

    base = 1 << basebit # base=2 in [CGGI16]
    prec_offset = 1 << (32 - (1 + basebit * t)) # precision
    mask = base - 1

    js = numpy.arange(1, t+1).reshape(1, 1, t)
    ai = ai.reshape(ai.shape + (1,))
    aijs = (((ai + prec_offset) >> (32 - js * basebit)) & mask) + 1

    for i in range(result.shape[0]):
        for l in range(n):
            for j in range(t):
                x = aijs[i,l,j] - 1
                if x != 0:
                    result.a[i,:] -= ks.a[l,j,x,:]
                    # FIXME: numpy detects overflow there, and gives a warning,
                    # but it's normal finite size integer arithmetic, and works as intended
                    result.b[i] -= ks.b[l,j,x]
                    result.current_variances[i] += ks.current_variances[l,j,x]


#sample=(a',b')
def lweKeySwitch(result: LweSampleArray, ks: LweKeySwitchKey, sample: LweSampleArray):

    params = ks.out_params
    n = ks.n
    basebit = ks.basebit
    t = ks.t

    lweNoiselessTrivial(result, sample.b, params)
    lweKeySwitchTranslate_fromArray(result, ks.ks, params, sample.a, n, t, basebit)

