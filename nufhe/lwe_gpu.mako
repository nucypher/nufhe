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

<%def name="make_lwe_keyswitch_key(
        idxs, ks_a, ks_b, ks_cv, in_key, b_term, noises_a, noises_b)">

    int input_idx = ${idxs[0]};
    int decomp_idx = ${idxs[1]};
    int base_idx = ${idxs[2]};

    if (base_idx == 0)
    {
        for (int i = 0; i < ${output_size}; i++)
        {
            ${ks_a.store_idx}(input_idx, decomp_idx, 0, i, 0);
        }
        ${ks_b.store_idx}(input_idx, decomp_idx, 0, 0);
        ${ks_cv.store_idx}(input_idx, decomp_idx, 0, 0);
    }
    else
    {
        int key = ${in_key.load_idx}(input_idx);
        int message = key * base_idx * (1 << (32 - (decomp_idx + 1) * ${log2_base}));

        ${noises_b.ctype} noise_b = ${noises_b.load_idx}(input_idx, decomp_idx, base_idx - 1);

        ${ks_cv.store_idx}(input_idx, decomp_idx, base_idx, ${noise**2});

        for (int i = 0; i < ${output_size}; i++)
        {
            ${ks_a.store_idx}(
                input_idx, decomp_idx, base_idx, i,
                ${noises_a.load_idx}(input_idx, decomp_idx, base_idx - 1, i));
        }

        ${ks_b.store_idx}(
            input_idx, decomp_idx, base_idx,
            message
            + noise_b
            + ${b_term.load_idx}(input_idx, decomp_idx, base_idx - 1));
    }
</%def>


<%def name="lwe_keyswitch(
    kernel_declaration, result_a, result_b, result_cv, ks_a, ks_b, ks_cv, source_a, source_b)">

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    const int decomp_mask = ${2**log2_base - 1};
    const int decomp_offset = ${2**(31 - decomp_length * log2_base)};

    unsigned int tid = virtual_local_id(1);
    unsigned int bdim = virtual_local_size(1);
    unsigned int batch_id = virtual_global_id(0);

    for (int i = tid; i < ${output_size}; i += bdim)
    {
        // Starting from a noiseless trivial LWE:
        // a = 0, b = source_b, current_variances = 0
        int res_a = 0;
        int res_b;
        double res_cv;

        if (i == 0)
        {
            res_b = ${source_b.load_combined_idx(slices[:-1])}(batch_id);
            res_cv = 0;
        }

        for (int j = 0; j < ${input_size}; j ++)
        {
            int tmp = ${source_a.load_combined_idx(slices)}(batch_id, j) + decomp_offset;

            for (int k = 0; k < ${decomp_length}; k++)
            {
                int val = (tmp >> (32 - (k + 1) * ${log2_base})) & decomp_mask;

                ## TODO: if val == 0, the corresponding keyswitch key slice is 0.
                ## Check if removing the condition increases the performance or not
                if (val != 0)
                    res_a -= ${ks_a.load_idx}(j, k, val, i);

                if (i == 0)
                {
                    if (val != 0)
                    {
                        res_b -= ${ks_b.load_idx}(j, k, val);
                        res_cv += ${ks_cv.load_idx}(j, k, val);
                    }
                }
            }
        }
        ${result_a.store_combined_idx(slices)}(batch_id, i, res_a);

        if (i == 0)
        {
            ${result_b.store_combined_idx(slices[:-1])}(batch_id, res_b);
            ${result_cv.store_combined_idx(slices[:-1])}(batch_id, res_cv);
        }
    }
}

</%def>


<%def name="lwe_linear(
    kernel_declaration, result_a, result_b, result_cv, source_a, source_b, source_cv, p)">

<%
    sshape = source_b.shape
    rshape = result_b.shape

    batch_ids = ["batch_id_" + str(i) for i in range(len(rshape))]
    result_ids = ", ".join(batch_ids)
    source_ids = ", ".join(
        "0" if sshape[i - (len(rshape) - len(sshape))] == 1 else batch_ids[i]
        for i in range(len(rshape) - len(sshape), len(rshape)))
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    %for i in range(len(rshape)):
    int ${batch_ids[i]} = virtual_global_id(${i});
    %endfor
    int n_id = virtual_global_id(${len(rshape)});

    ${result_a.store_idx}(
        ${result_ids}, n_id,
        %if add_result:
        ${result_a.load_idx}(${result_ids}, n_id)
        %endif
        + ${p} * ${source_a.load_idx}(${source_ids}, n_id));

    if (n_id == 0)
    {
        ${result_b.store_idx}(
            ${result_ids},
            %if add_result:
            ${result_b.load_idx}(${result_ids})
            %endif
            + ${p} * ${source_b.load_idx}(${source_ids}));
        ${result_cv.store_idx}(
            ${result_ids},
            %if add_result:
            ${result_cv.load_idx}(${result_ids})
            %endif
            + ${p} * ${p} * ${source_cv.load_idx}(${source_ids}));
    }
}
</%def>


<%def name="lwe_noiseless_trivial(
    kernel_declaration, result_a, result_b, result_cv, mus)">

<%
    sshape = mus.shape
    rshape = result_b.shape

    batch_ids = ["batch_id_" + str(i) for i in range(len(rshape))]
    result_ids = ", ".join(batch_ids)
    source_ids = ", ".join(
        "0" if sshape[i - (len(rshape) - len(sshape))] == 1 else batch_ids[i]
        for i in range(len(rshape) - len(sshape), len(rshape)))
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    %for i in range(len(rshape)):
    int ${batch_ids[i]} = virtual_global_id(${i});
    %endfor
    int n_id = virtual_global_id(${len(rshape)});

    ${result_a.store_idx}(${result_ids}, n_id, 0);
    if (n_id == 0)
    {
        ${result_b.store_idx}(${result_ids}, ${mus.load_idx}(${source_ids}));
        ${result_cv.store_idx}(${result_ids}, 0);
    }
}
</%def>
