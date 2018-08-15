## Copyright (C) 2018 NuCypher
##
## This file is part of nufhe.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

<%def name="tlwe_encrypt_zero_fill_result(
    kernel_declaration, result_a, result_cv, noises1, noises2, ift_res)">

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    int batch_idx = virtual_global_id(0);
    int mask_idx = virtual_global_id(1);
    int poly_idx = virtual_global_id(2);

    int result_a;

    if (mask_idx < ${mask_size})
    {
        result_a = ${noises1.load_combined_idx(noises1_slices)}(batch_idx, mask_idx, poly_idx);
    }
    else
    {
        result_a = ${noises2.load_combined_idx(noises2_slices)}(batch_idx, poly_idx);
        for (int i = 0; i < ${mask_size}; i++)
        {
            result_a += ${ift_res.load_combined_idx(noises1_slices)}(batch_idx, i, poly_idx);
        }
    }
    ${result_a.store_combined_idx(noises1_slices)}(batch_idx, mask_idx, poly_idx, result_a);


    if (poly_idx == 0 && mask_idx == 0)
    {
        ${result_cv.store_combined_idx(cv_slices)}(batch_idx, ${noise**2});
    }
}
</%def>


<%def name="tlwe_extract_lwe_samples(kernel_declaration, result_a, result_b, tlwe_a)">
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    int batch_idx = virtual_global_id(0);
    int mask_idx = virtual_global_id(1);
    int poly_idx = virtual_global_id(2);

    ${result_a.ctype} a;
    if (poly_idx == 0)
    {
        a = ${tlwe_a.load_combined_idx(slices)}(
            batch_idx, mask_idx, 0);
    }
    else
    {
        a = -${tlwe_a.load_combined_idx(slices)}(
            batch_idx, mask_idx, ${polynomial_degree} - poly_idx);
    }
    ${result_a.store_combined_idx(slices[:-1])}(
        batch_idx, mask_idx * ${polynomial_degree} + poly_idx, a);

    if (mask_idx == 0 && poly_idx == 0)
    {
        ${result_b.store_combined_idx(slices[:-2])}(
            batch_idx,
            ${tlwe_a.load_combined_idx(slices)}(batch_idx, ${mask_size}, 0));
    }
}
</%def>
