import numpy

from .lwe import *
from .tgsw import *


def lwe_bootstrapping_key(
        rng, ks_t: int, ks_basebit: int, key_in: LweKey, rgsw_key: TGswKey):

    bk_params = rgsw_key.params
    in_out_params = key_in.params
    accum_params = bk_params.tlwe_params
    extract_params = accum_params.extracted_lweparams

    n = in_out_params.n
    N = extract_params.n

    accum_key = rgsw_key.tlwe_key
    extracted_key = LweKey.from_key(extract_params, accum_key)

    ks = LweKeySwitchKey(rng, N, ks_t, ks_basebit, extracted_key, key_in)

    bk = TGswSampleArray(bk_params, (n,))
    kin = key_in.key
    alpha = accum_params.alpha_min

    tGswSymEncryptInt(rng, bk, kin, alpha, rgsw_key)

    return bk, ks


class LweBootstrappingKeyFFT:

    def __init__(self, rng, ks_t: int, ks_basebit: int, lwe_key: LweKey, tgsw_key: TGswKey):
        in_out_params = lwe_key.params
        bk_params = tgsw_key.params
        accum_params = bk_params.tlwe_params
        extract_params = accum_params.extracted_lweparams

        bk, ks = lwe_bootstrapping_key(rng, ks_t, ks_basebit, lwe_key, tgsw_key)

        n = in_out_params.n

        # Bootstrapping Key FFT
        bkFFT = TGswSampleFFTArray(bk_params, (n,))
        tGswToFFTConvert(bkFFT, bk, bk_params)

        self.in_out_params = in_out_params # paramètre de l'input et de l'output. key: s
        self.bk_params = bk_params # params of the Gsw elems in bk. key: s"
        self.accum_params = accum_params # params of the accum variable key: s"
        self.extract_params = extract_params # params after extraction: key: s'
        self.bkFFT = bkFFT # the bootstrapping key (s->s")
        self.ks = ks # the keyswitch key (s'->s)


def tfhe_MuxRotate_FFT(
        result: TLweSampleArray, accum: TLweSampleArray, bki: TGswSampleFFTArray, bk_idx: int,
        barai, bk_params: TGswParams, tmpa: TLweSampleFFTArray,
        deca: IntPolynomialArray, decaFFT: LagrangeHalfCPolynomialArray):

    # TYPING: barai::Array{Int32}
    # ACC = BKi*[(X^barai-1)*ACC]+ACC
    # temp = (X^barai-1)*ACC
    tLweMulByXaiMinusOne(result, barai, accum, bk_params.tlwe_params)

    # temp *= BKi
    tGswFFTExternMulToTLwe(result, bki, bk_idx, bk_params, tmpa, deca, decaFFT)

    # ACC += temp
    tLweAddTo(result, accum, bk_params.tlwe_params)


"""
 * multiply the accumulator by X^sum(bara_i.s_i)
 * @param accum the TLWE sample to multiply
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
"""
def tfhe_blindRotate_FFT(
        accum: TLweSampleArray, bkFFT: TGswSampleFFTArray, bara, n: int, bk_params: TGswParams):

    # TYPING: bara::Array{Int32}
    temp = TLweSampleArray(bk_params.tlwe_params, accum.shape)
    temp2 = temp
    temp3 = accum

    accum_in_temp3 = True

    # For use in tGswFFTExternMulToTLwe(), so that we don't have to allocate them `n` times
    tmpa = TLweSampleFFTArray(bk_params.tlwe_params, accum.shape)
    deca = IntPolynomialArray(bk_params.tlwe_params.N, accum.a.shape + (bk_params.l,))
    decaFFT = LagrangeHalfCPolynomialArray(
        bk_params.tlwe_params.N, accum.shape + (bk_params.tlwe_params.k + 1, bk_params.l))

    for i in range(n):
        # GPU: will have to be passed as a pair `bara`, `i`
        barai = bara[:,i] # !!! assuming the ciphertext is 1D

        # FIXME: We could pass the view bkFFT[i] here, but on the current Julia it's too slow
        tfhe_MuxRotate_FFT(temp2, temp3, bkFFT, i, barai, bk_params, tmpa, deca, decaFFT)

        temp2, temp3 = temp3, temp2
        accum_in_temp3 = not accum_in_temp3

    if not accum_in_temp3: # temp3 != accum
        tLweCopy(accum, temp3, bk_params.tlwe_params)


"""
 * result = LWE(v_p) where p=barb-sum(bara_i.s_i) mod 2N
 * @param result the output LWE sample
 * @param v a 2N-elt anticyclic function (represented by a TorusPolynomial)
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param barb A coefficients between 0 and 2N-1
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
"""
def tfhe_blindRotateAndExtract_FFT(
        result: LweSampleArray,
        v: TorusPolynomialArray, bk: TGswSampleFFTArray, barb, bara, n: int, bk_params: TGswParams):

    # TYPING: barb::Array{Int32},
    # TYPING: bara::Array{Int32}

    accum_params = bk_params.tlwe_params
    extract_params = accum_params.extracted_lweparams
    N = accum_params.N

    # Test polynomial
    testvectbis = TorusPolynomialArray(N, result.shape)
    # Accumulator
    acc = TLweSampleArray(accum_params, result.shape)

    # testvector = X^{2N-barb}*v
    # GPU: array operations or a custom kernel
    tp_mul_by_xai_(testvectbis, 2 * N - barb, v)

    tLweNoiselessTrivial(acc, testvectbis, accum_params)

    # Blind rotation
    tfhe_blindRotate_FFT(acc, bk, bara, n, bk_params)

    # Extraction
    tLweExtractLweSample(result, acc, extract_params, accum_params)


"""
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
"""
def tfhe_bootstrap_woKS_FFT(
        result: LweSampleArray, bk: LweBootstrappingKeyFFT, mu: Torus32, x: LweSampleArray):

    bk_params = bk.bk_params
    accum_params = bk.accum_params
    in_params = bk.in_out_params
    N = accum_params.N
    n = in_params.n

    testvect = TorusPolynomialArray(N, result.shape)

    # Modulus switching
    # GPU: array operations or a custom kernel
    barb = modSwitchFromTorus32(x.b, 2 * N)
    bara = modSwitchFromTorus32(x.a, 2 * N)

    # the initial testvec = [mu,mu,mu,...,mu]
    # TODO: use an appropriate method
    # GPU: array operations or a custom kernel
    testvect.coefsT.fill(mu)

    # Bootstrapping rotation and extraction
    tfhe_blindRotateAndExtract_FFT(result, testvect, bk.bkFFT, barb, bara, n, bk_params)


"""
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
"""
def tfhe_bootstrap_FFT(
        result: LweSampleArray, bk: LweBootstrappingKeyFFT, mu: Torus32, x: LweSampleArray):

    u = LweSampleArray(bk.accum_params.extracted_lweparams, result.shape)

    tfhe_bootstrap_woKS_FFT(u, bk, mu, x)

    # Key switching
    lweKeySwitch(result, bk.ks, u)
