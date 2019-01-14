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

from .numeric_functions import Torus32, t32_to_phase
from .polynomials import TorusPolynomialArray, shift_tp_inverted_power
from .lwe import LweParams, LweKey, LweSampleArray, LweKeyswitchKey, lwe_keyswitch
from .tgsw import (
    TGswKey,
    TransformedTGswSampleArray,
    TGswParams,
    TGswSampleArray,
    tgsw_transform_samples,
    tgsw_encrypt_int,
    tgsw_transformed_external_mul,
    )
from .tlwe import (
    TLweSampleArray,
    tlwe_noiseless_trivial,
    tlwe_shift_polynomials,
    tlwe_add_to,
    tlwe_extract_lwe_samples,
    tlwe_copy,
    )
from .blind_rotate import BlindRotate_gpu
from .performance import PerformanceParametersForDevice


class BootstrapKey:

    def __init__(
            self, in_out_params: LweParams, tgsw: TransformedTGswSampleArray):

        bk_params = tgsw.params
        accum_params = bk_params.tlwe_params
        extract_params = accum_params.extracted_lweparams

        self.in_out_params = in_out_params
        self.bk_params = bk_params
        self.accum_params = accum_params
        self.extract_params = extract_params
        self.tgsw = tgsw

    @classmethod
    def from_rng(
            cls, thr, rng, lwe_key: LweKey, tgsw_key: TGswKey,
            perf_params: PerformanceParametersForDevice):

        in_out_params = lwe_key.params
        bk_params = tgsw_key.params
        accum_params = bk_params.tlwe_params

        # Make a non-transformed bootstrap key
        bk = TGswSampleArray.empty(thr, bk_params, (in_out_params.size,))
        tgsw_encrypt_int(thr, rng, bk, lwe_key.key, accum_params.min_noise, tgsw_key, perf_params)

        # Convert it to transformed space, because that's where it will be used
        bk_transformed = TransformedTGswSampleArray.empty(thr, bk_params, (in_out_params.size,))
        tgsw_transform_samples(thr, bk_transformed, bk, perf_params)

        return cls(in_out_params, bk_transformed)

    def dump(self, file_obj):
        pickle.dump(self.in_out_params, file_obj)
        self.tgsw.dump(file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        in_out_params = pickle.load(file_obj)
        tgsw = TransformedTGswSampleArray.load(file_obj, thr)
        return cls(in_out_params, tgsw)

    def __eq__(self, other: 'BootstrapKey'):
        return (
            self.__class__ == other.__class__
            and self.in_out_params == other.in_out_params
            and self.tgsw == other.tgsw)



def mux_rotate(
        thr, result: TLweSampleArray, accum: TLweSampleArray, bki: TransformedTGswSampleArray,
        bk_idx: int, barai, bk_params: TGswParams, perf_params: PerformanceParametersForDevice):

    # TYPING: barai::Array{Int32}
    # ACC = BKi*[(X^barai-1)*ACC]+ACC
    # temp = (X^barai-1)*ACC
    tlwe_shift_polynomials(thr, result, accum, barai, bk_idx)

    # temp *= BKi
    tgsw_transformed_external_mul(thr, result, bki, bk_idx, perf_params)

    # ACC += temp
    tlwe_add_to(thr, result, accum)


"""
 * multiply the accumulator by X^sum(bara_i.s_i)
 * @param accum the TLWE sample to multiply
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
"""
def blind_rotate(
        thr, accum: TLweSampleArray, bk: BootstrapKey,
        bara, n: int, bk_params: TGswParams, perf_params: PerformanceParametersForDevice):

    # TYPING: bara::Array{Int32}
    temp = TLweSampleArray.empty(thr, bk_params.tlwe_params, accum.shape)

    temp2 = temp
    temp3 = accum

    accum_in_temp3 = True

    for i in range(n):
        # TODO: here we only need to pass bootstrap_key[i] and bara[:,i],
        # but Reikna kernels have to be recompiled for every set of strides/offsets,
        # so for now we are just passing full arrays and an index.
        mux_rotate(thr, temp2, temp3, bk.tgsw, i, bara, bk_params, perf_params)

        temp2, temp3 = temp3, temp2
        accum_in_temp3 = not accum_in_temp3

    # TODO: add a test that checks this
    if not accum_in_temp3: # temp3 != accum
        tlwe_copy(thr, accum, temp3)


"""
 * result = LWE(v_p) where p=barb-sum(bara_i.s_i) mod 2N
 * @param result the output LWE sample
 * @param v a 2N-elt anticyclic function (represented by a TorusPolynomial)
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param barb A coefficients between 0 and 2N-1
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
"""
def blind_rotate_and_extract(
        thr, result: LweSampleArray,
        v: TorusPolynomialArray, bk: BootstrapKey, ks: LweKeyswitchKey,
        barb, bara,
        perf_params: PerformanceParametersForDevice,
        no_keyswitch=False):

    # TYPING: barb::Array{Int32},
    # TYPING: bara::Array{Int32}

    bk_params = bk.bk_params

    if not no_keyswitch:
        extracted_result = LweSampleArray.empty(
            thr, bk.accum_params.extracted_lweparams, result.shape_info.shape)
    else:
        extracted_result = result

    accum_params = bk_params.tlwe_params
    extract_params = accum_params.extracted_lweparams
    N = accum_params.polynomial_degree

    # testvector = X^{2N-barb}*v
    testvectbis = TorusPolynomialArray.empty(thr, N, extracted_result.shape_info.shape)
    shift_tp_inverted_power(thr, testvectbis, barb, v)

    # Accumulator
    acc = TLweSampleArray.empty(thr, accum_params, extracted_result.shape_info.shape)
    tlwe_noiseless_trivial(thr, acc, testvectbis)

    if perf_params.single_kernel_bootstrap:
        # includes blindrotate, extractlwesample and (optionally) keyswitch
        BlindRotate_gpu(result, acc, bk, ks, bara, perf_params, no_keyswitch=no_keyswitch)
    else:
        # Blind rotation
        blind_rotate(
            thr, acc, bk, bara, bk.in_out_params.size, bk_params, perf_params)

        # Extraction
        tlwe_extract_lwe_samples(thr, extracted_result, acc)

        if not no_keyswitch:
            lwe_keyswitch(thr, result, ks, extracted_result)


"""
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
"""
def bootstrap(
        thr, result: LweSampleArray, bk: BootstrapKey, ks: LweKeyswitchKey,
        mu: Torus32, x: LweSampleArray, perf_params: PerformanceParametersForDevice,
        no_keyswitch=False):

    accum_params = bk.accum_params
    N = accum_params.polynomial_degree

    testvect = TorusPolynomialArray.empty(thr, N, result.shape_info.shape)

    # Modulus switching
    barb = thr.array(x.b.shape, Torus32)
    bara = thr.array(x.a.shape, Torus32)

    t32_to_phase(thr, barb, x.b, 2 * N)
    t32_to_phase(thr, bara, x.a, 2 * N)

    # the initial testvec = [mu,mu,mu,...,mu]
    testvect.coeffs.fill(mu)

    # Bootstrapping rotation and extraction
    blind_rotate_and_extract(
        thr, result, testvect, bk, ks, barb, bara, perf_params,
        no_keyswitch=no_keyswitch)
