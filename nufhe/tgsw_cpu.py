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

from reikna import helpers

from .numeric_functions import Torus32
from .polynomial_transform import get_transform


def tgsw_polynomial_decomp_trf_reference(params: 'TGswParams', shape):

    tlwe_params = params.tlwe_params
    decomp_length = params.decomp_length
    mask_size = tlwe_params.mask_size
    polynomial_degree = tlwe_params.polynomial_degree

    bs_log2_base = params.bs_log2_base
    offset = params.offset
    base = 2**bs_log2_base
    maskMod = base - 1
    halfBg = 2**(bs_log2_base - 1)

    def _kernel(result, sample):

        decomp_shift = lambda p: 32 - p * bs_log2_base

        ps = numpy.arange(1, decomp_length + 1).reshape((1,) * len(shape) + (1, decomp_length, 1))
        sample_coefs = sample.reshape(shape + (mask_size + 1, 1, polynomial_degree))

        # do the decomposition
        numpy.copyto(result, (((sample_coefs + offset) >> decomp_shift(ps)) & maskMod) - halfBg)

    return _kernel


def tlwe_transformed_add_mul_to_trf_reference(
        params: 'TGswParams', shape, bk_len: int, perf_params):

    tlwe_params = params.tlwe_params
    decomp_length = params.decomp_length
    mask_size = tlwe_params.mask_size
    polynomial_degree = tlwe_params.polynomial_degree

    transform = get_transform(tlwe_params.transform_type)
    tlength = transform.transformed_length(polynomial_degree)

    def _kernel(result, sample, bootstrap_key, bk_row_idx):
        batch_len = helpers.product(shape)
        sample_view = sample.reshape(batch_len, mask_size + 1, decomp_length, 1, tlength)
        result_view = result.reshape(batch_len, mask_size + 1, tlength)

        result.fill(0)
        for mask_idx in range(mask_size + 1):
            for decomp_idx in range(decomp_length):
                numpy.copyto(
                    result_view,
                    transform.transformed_space_add_ref(
                        result_view,
                        transform.transformed_space_mul_prepared_ref(
                            sample_view[:, mask_idx, decomp_idx, :, :],
                            bootstrap_key[bk_row_idx, mask_idx, decomp_idx, :, :])))

    return _kernel


def TGswTransformedExternalMulReference(params: 'TGswParams', shape, bk_len, perf_params):

    tlwe_params = params.tlwe_params
    decomp_length = params.decomp_length
    mask_size = tlwe_params.mask_size
    polynomial_degree = tlwe_params.polynomial_degree

    transform = get_transform(params.tlwe_params.transform_type)
    tlength = transform.transformed_length(polynomial_degree)
    tdtype = transform.transformed_dtype()

    def _kernel(accum, bootstrap_key, bk_row_idx):

        sample = numpy.empty(shape + (mask_size + 1, decomp_length, polynomial_degree), Torus32)
        tr_accum = numpy.empty(shape + (mask_size + 1, tlength), tdtype)

        decomp = tgsw_polynomial_decomp_trf_reference(params, shape)
        add_mul = tlwe_transformed_add_mul_to_trf_reference(params, shape, bk_len, perf_params)

        decomp(sample, accum)
        tr_sample = transform.forward_transform_ref(sample)
        add_mul(tr_accum, tr_sample, bootstrap_key, bk_row_idx)
        numpy.copyto(accum, transform.inverse_transform_ref(tr_accum))

    return _kernel


def TGswAddMessageReference(params: 'TGswParams', shape):
    mask_size = params.tlwe_params.mask_size
    polynomial_degree = params.tlwe_params.polynomial_degree
    decomp_length = params.decomp_length
    base_powers = params.base_powers

    def _kernel(result_a, messages):
        result_a_view = result_a.reshape(
            helpers.product(messages.shape),
            mask_size + 1, decomp_length, mask_size + 1, polynomial_degree)
        messages = messages.flatten()

        inc = messages.reshape(messages.size, 1) * base_powers.reshape(1, decomp_length)

        for mask_idx in range(mask_size + 1):
            result_a_view[:, mask_idx, :, mask_idx, 0] += inc

    return _kernel
