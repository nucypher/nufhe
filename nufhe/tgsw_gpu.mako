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

<%def name="tgsw_add_message(kernel_declaration, result_a, messages)">

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    int batch_idx = virtual_global_id(0);

    %for mask_idx in range(mask_size + 1):
    %for decomp_idx in range(decomp_length):
    {
    ${result_a.ctype} result_a = ${result_a.load_combined_idx(slices)}(
        batch_idx, ${mask_idx}, ${decomp_idx}, ${mask_idx}, 0);
    ${messages.ctype} message = ${messages.load_combined_idx(slices[:-4])}(batch_idx);
    ${result_a.store_combined_idx(slices)}(
        batch_idx, ${mask_idx}, ${decomp_idx}, ${mask_idx}, 0,
        result_a + message * (${base_powers[decomp_idx]}));
    }
    %endfor
    %endfor
}
</%def>
