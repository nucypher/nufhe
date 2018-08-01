from .numeric_functions import Torus32
from .gpu_polynomials import TorusPolynomialArray
from .lwe import LweKey, LweSampleArray, LweKeyswitchKey, keyswitch
from .tgsw import TGswKey, TGswSampleFFTArray, TGswParams, TGswSampleArray, tGswToFFTConvert
from .tgsw_gpu import tGswSymEncryptInt_gpu, tGswFFTExternMulToTLwe_gpu
from .tlwe import TLweSampleArray
from .tlwe_gpu import (
    tLweNoiselessTrivial_gpu,
    tLweMulByXaiMinusOne_gpu,
    tLweAddTo_gpu,
    tLweExtractLweSample_gpu,
    )
from .numeric_functions_gpu import modSwitchFromTorus32_gpu
from .blind_rotate import BlindRotate_gpu
from .gpu_polynomials import tp_mul_by_xai_gpu
from .performance import PerformanceParameters

import time

to_gpu_time = 0


def lwe_bootstrapping_key(
        thr, rng, ks_decomp_length: int, ks_log2_base: int, key_in: LweKey, rgsw_key: TGswKey,
        perf_params: PerformanceParameters):

    bk_params = rgsw_key.params
    in_out_params = key_in.params
    accum_params = bk_params.tlwe_params
    extract_params = accum_params.extracted_lweparams

    accum_key = rgsw_key.tlwe_key
    extracted_key = LweKey.from_tlwe_key(extract_params, accum_key)

    ks = LweKeyswitchKey(thr, rng, extracted_key, key_in, ks_decomp_length, ks_log2_base)

    bk = TGswSampleArray(thr, bk_params, (in_out_params.size,))
    kin = key_in.key
    noise = accum_params.min_noise

    tGswSymEncryptInt_gpu(thr, rng, bk, kin, noise, rgsw_key, perf_params)

    return bk, ks


class LweBootstrappingKeyFFT:

    def __init__(
            self, thr, rng, ks_decomp_length: int, ks_log2_base: int,
            lwe_key: LweKey, tgsw_key: TGswKey, perf_params: PerformanceParameters):

        in_out_params = lwe_key.params
        bk_params = tgsw_key.params
        accum_params = bk_params.tlwe_params
        extract_params = accum_params.extracted_lweparams

        bk, ks = lwe_bootstrapping_key(
            thr, rng, ks_decomp_length, ks_log2_base, lwe_key, tgsw_key, perf_params)

        n = in_out_params.size

        # Bootstrapping Key FFT
        bkFFT = TGswSampleFFTArray(thr, bk_params, (n,))
        tGswToFFTConvert(thr, bkFFT, bk, bk_params, perf_params)

        self.in_out_params = in_out_params # paramètre de l'input et de l'output. key: s
        self.bk_params = bk_params # params of the Gsw elems in bk. key: s"
        self.accum_params = accum_params # params of the accum variable key: s"
        self.extract_params = extract_params # params after extraction: key: s'
        self.bkFFT = bkFFT # the bootstrapping key (s->s")
        self.ks = ks # the keyswitch key (s'->s)


def tfhe_MuxRotate_FFT(
        result: TLweSampleArray, accum: TLweSampleArray, bki: TGswSampleFFTArray, bk_idx: int,
        barai, bk_params: TGswParams, perf_params: PerformanceParameters):

    # TYPING: barai::Array{Int32}
    # ACC = BKi*[(X^barai-1)*ACC]+ACC
    # temp = (X^barai-1)*ACC
    tLweMulByXaiMinusOne_gpu(result, barai, bk_idx, accum, bk_params.tlwe_params)

    # temp *= BKi
    tGswFFTExternMulToTLwe_gpu(result, bki, bk_idx, bk_params, perf_params)

    # ACC += temp
    tLweAddTo_gpu(result, accum, bk_params.tlwe_params)


"""
 * multiply the accumulator by X^sum(bara_i.s_i)
 * @param accum the TLWE sample to multiply
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
"""
def tfhe_blindRotate_FFT(
        accum: TLweSampleArray, bkFFT: TGswSampleFFTArray, bara, n: int, bk_params: TGswParams,
        perf_params: PerformanceParameters):

    thr = accum.a.coefsT.thread

    global to_gpu_time

    # TYPING: bara::Array{Int32}
    t = time.time()
    thr.synchronize()
    temp = TLweSampleArray(thr, bk_params.tlwe_params, accum.shape)
    thr.synchronize()
    to_gpu_time += time.time() - t

    temp2 = temp
    temp3 = accum

    accum_in_temp3 = True

    for i in range(n):
        # TODO: here we only need to pass bkFFT[i] and bara[:,i],
        # but Reikna kernels have to be recompiled for every set of strides/offsets,
        # so for now we are just passing full arrays and an index.
        tfhe_MuxRotate_FFT(temp2, temp3, bkFFT, i, bara, bk_params, perf_params)

        temp2, temp3 = temp3, temp2
        accum_in_temp3 = not accum_in_temp3

    if not accum_in_temp3: # temp3 != accum
        tLweCopy_gpu(accum, temp3, bk_params.tlwe_params)


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
        thr, result: LweSampleArray,
        v: TorusPolynomialArray, bk: LweBootstrappingKeyFFT,
        barb, bara,
        perf_params: PerformanceParameters,
        no_keyswitch=False):

    # TYPING: barb::Array{Int32},
    # TYPING: bara::Array{Int32}

    global to_gpu_time

    bk_params = bk.bk_params

    if not no_keyswitch:
        t = time.time()
        extracted_result = LweSampleArray.empty(
            thr, bk.accum_params.extracted_lweparams, result.shape_info.shape)
        thr.synchronize()
        to_gpu_time += time.time() - t
    else:
        extracted_result = result

    accum_params = bk_params.tlwe_params
    extract_params = accum_params.extracted_lweparams
    N = accum_params.polynomial_degree

    # Test polynomial
    t = time.time()
    thr.synchronize()
    testvectbis = TorusPolynomialArray(thr, N, extracted_result.shape_info.shape)

    # Accumulator
    acc = TLweSampleArray(thr, accum_params, extracted_result.shape_info.shape)
    thr.synchronize()
    to_gpu_time += time.time() - t

    # testvector = X^{2N-barb}*v
    tp_mul_by_xai_gpu(testvectbis, barb, v, invert_ais=True)

    tLweNoiselessTrivial_gpu(acc, testvectbis, accum_params)

    if perf_params.single_kernel_bootstrap:
        # includes blindrotate, extractlwesample and (optionally) keyswitch
        BlindRotate_gpu(result, acc, bk, bara, perf_params, no_keyswitch=no_keyswitch)

    else:
        # Blind rotation
        tfhe_blindRotate_FFT(acc, bk.bkFFT, bara, bk.in_out_params.size, bk_params, perf_params)

        # Extraction
        tLweExtractLweSample_gpu(extracted_result, acc, extract_params, accum_params)

        if not no_keyswitch:
            keyswitch(thr, result, bk.ks, extracted_result)


"""
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
"""
def bootstrap(
        thr, result: LweSampleArray, bk: LweBootstrappingKeyFFT, mu: Torus32, x: LweSampleArray,
        perf_params: PerformanceParameters,
        no_keyswitch=False):

    accum_params = bk.accum_params
    N = accum_params.polynomial_degree

    global to_gpu_time

    thr = result.a.thread
    t = time.time()
    thr.synchronize()
    testvect = TorusPolynomialArray(thr, N, result.shape_info.shape)
    thr.synchronize()
    to_gpu_time += time.time() - t

    # Modulus switching
    barb = thr.array(x.b.shape, Torus32)
    bara = thr.array(x.a.shape, Torus32)

    modSwitchFromTorus32_gpu(barb, x.b, 2 * N)
    modSwitchFromTorus32_gpu(bara, x.a, 2 * N)

    # the initial testvec = [mu,mu,mu,...,mu]
    testvect.coefsT.fill(mu)

    # Bootstrapping rotation and extraction
    tfhe_blindRotateAndExtract_FFT(
        thr, result, testvect, bk, barb, bara, perf_params,
        no_keyswitch=no_keyswitch)
