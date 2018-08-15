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

<%def name="shift_torus_polynomial(kernel_declaration, result, source, powers, powers_idx)">
<%
    poly_slices = (batch_len, poly_batch_len, 1)
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    int batch_id = virtual_global_id(0);
    int poly_batch_id = virtual_global_id(1);
    int polynomial_id = virtual_global_id(2);

    %if powers_view:
    ${powers.ctype} power = ${powers.load_combined_idx((batch_len, 1))}(batch_id, ${powers_idx});
    %else:
    ${powers.ctype} power = ${powers.load_combined_idx((batch_len,))}(batch_id);
    %endif

    %if invert_powers:
    power = ${2 * polynomial_degree} - power;
    %endif

    ${result.ctype} res;

    if (power < ${polynomial_degree})
    {
        if (polynomial_id < power)
        {
            res = -${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id + ${polynomial_degree} - power);
        }
        else
        {
            res = ${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id - power);
        }
    }
    else
    {
        power = power - ${polynomial_degree};
        if (polynomial_id < power)
        {
            res = ${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id + ${polynomial_degree} - power);
        }
        else
        {
            res = -${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id - power);
        }
    }

    %if minus_one:
    res -= ${source.load_combined_idx(poly_slices)}(batch_id, poly_batch_id, polynomial_id);
    %endif

    ${result.store_combined_idx(poly_slices)}(batch_id, poly_batch_id, polynomial_id, res);
}
</%def>
