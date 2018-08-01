import numpy

from .numeric_functions import Torus32, dtot32


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
def LweKeySwitchTranslate_fromArray_reference(
        shape_info, t: int, outer_n, inner_n, basebit: int):

    def _kernel(a, b, current_variances, ks_a, ks_b, ks_current_variances, ai, bi):

        base = 1 << basebit # base=2 in [CGGI16]
        prec_offset = 1 << (32 - (1 + basebit * t)) # precision
        mask = base - 1

        js = numpy.arange(1, t+1).reshape(1, 1, t)
        ai = ai.reshape(ai.shape + (1,))
        aijs = (((ai + prec_offset) >> (32 - js * basebit)) & mask)

        # Starting from a noiseless trivial LWE:
        # a = 0, b = bi, current_variances = 0
        a.fill(0)
        b[:] = bi
        current_variances.fill(0)

        for i in range(shape_info.shape[0]):
            for l in range(outer_n):
                for j in range(t):
                    x = aijs[i,l,j]
                    if x != 0:
                        a[i,:] -= ks_a[l,j,x,:]
                        b[i] -= ks_b[l,j,x]
                        current_variances[i] += ks_current_variances[l,j,x]

    return _kernel


def vec_mul_mat(b, a):
    return (a * b).sum(-1, dtype=numpy.int32)


# This function encrypts a message by using key and a given noise value
def lweSymEncryptWithExternalNoise(
        ks_a, ks_b, ks_cv, messages, a_noises, b_noises, alpha: float, key):

    # term h=0 as trivial encryption of 0 (it will not be used in the KeySwitching)
    ks_a[:,:,0,:] = 0
    ks_b[:,:,0] = 0
    ks_cv[:,:,0] = 0

    ks_b[:,:,1:] = messages + dtot32(b_noises)
    ks_a[:,:,1:,:] = a_noises
    ks_b[:,:,1:] += vec_mul_mat(key, a_noises)
    ks_cv[:,:,1:] = alpha**2


def LweKeySwitchKeyComputation_ref(extracted_n: int, t: int, basebit: int, inner_n: int, alpha):

    base = 1 << basebit

    def _kernel(ks_a, ks_b, ks_cv, in_key, out_key, a_noises, b_noises):

        # recenter the noises
        b_noises -= b_noises.mean()

        # generate the ks

        # mess::Torus32 = (in_key.key[i] * Int32(h - 1)) * Int32(1 << (32 - j * basebit))
        hs = numpy.arange(2, base+1)
        js = numpy.arange(1, t+1)

        r_key = in_key.reshape(extracted_n, 1, 1)
        r_hs = hs.reshape(1, 1, base - 1)
        r_js = js.reshape(1, t, 1)

        messages = r_key * (r_hs - 1) * (1 << (32 - r_js * basebit))
        messages = messages.astype(Torus32)

        lweSymEncryptWithExternalNoise(ks_a, ks_b, ks_cv, messages, a_noises, b_noises, alpha, out_key)

    return _kernel


# * This function encrypts message by using key, with stdev alpha
# * The Lwe sample for the result must be allocated and initialized
# * (this means that the parameters are already in the result)
def LweSymEncrypt_ref(shape, n, alpha: float):

    def _kernel(result_a, result_b, result_cv, messages, key, noises_a, noises_b):
        numpy.copyto(result_b, noises_b + messages)
        numpy.copyto(result_a, noises_a)
        result_b += vec_mul_mat(key, result_a)
        result_cv.fill(alpha**2)

    return _kernel


# This function computes the phase of sample by using key : phi = b - a.s
def LwePhase_ref(shape, n):

    def _kernel(result, a, b, key):
        numpy.copyto(result, b - vec_mul_mat(key, a))

    return _kernel


def LweLinear_ref(result_shape_info, source_shape_info, params, add_result=False):

    def _kernel(result_a, result_b, result_cv, source_a, source_b, source_cv, p):
        p = numpy.int32(p)
        numpy.copyto(result_a, (result_a if add_result else 0) + p * source_a)
        numpy.copyto(result_b, (result_b if add_result else 0) + p * source_b)
        numpy.copyto(result_cv, (result_cv if add_result else 0) + p**2 * source_cv)

    return _kernel
