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

<%def name="blind_rotate(
    kernel_declaration, extracted_a, extracted_b, accum_a, gsw, bara, cdata_forward, cdata_inverse)">
<%
    tpt = transform.threads_per_transform
    plength = transform.polynomial_length
    log2_plength = plength.bit_length() - 1

    p_ept = plength // transform.threads_per_transform
    tr_ept = transform.transform_length // transform.threads_per_transform

    # Makes the code a bit simpler, can be lifted if necessary
    assert plength % tpt == 0
    assert transform.transform_length % tpt == 0
    assert transform.cdata_fw.size % tpt == 0
    assert transform.cdata_inv.size % tpt == 0

    temp_ctype = dtypes.ctype(transform.temp_dtype)

    tr_size = transform.transform_length * transform.elem_dtype.itemsize
    temp_size = transform.temp_length * transform.temp_dtype.itemsize
    sh_size = max(tr_size, temp_size)
    sh_length_tr = sh_size // transform.elem_dtype.itemsize

    decomp_mask = 2**bs_log2_base - 1
    decomp_half = 2**(bs_log2_base - 1)
    decomp_offset = 2**31 + 2**(31 - bs_log2_base)
%>


${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    // We are trying to minimize the number of local memory buffers used.
    // We need `decomp_length * (mask_size + 1)` buffers to store the transformed data,
    // then all of them are used to fill `(mask_size + 1)` output buffers.
    // `mask_size` of them should be put outside of the input buffers,
    // but the last one can overwrite one of the input buffers.
    //
    // So, for example, for `mask_size=1` and `decomp_length=2`, we need
    // `(1 + 1) * 2 = 4` input buffers and one additional output buffer.
    LOCAL_MEM char sh_char[${sh_size * ((mask_size + 1) * decomp_length + mask_size)}];
    LOCAL_MEM ${accum_a.ctype} shared_accum[${(mask_size + 1) * plength}];

    LOCAL_MEM_ARG ${tr_ctype}* sh = (LOCAL_MEM_ARG ${tr_ctype}*)sh_char;

    const unsigned int batch_id = virtual_group_id(0);
    const unsigned int tid = virtual_local_id(1);
    const unsigned int transform_id = tid / ${transform.threads_per_transform};
    const unsigned int mask_id = transform_id % ${mask_size + 1};
    const unsigned int decomp_id = transform_id / ${mask_size + 1};
    const unsigned int thread_in_transform = tid % ${transform.threads_per_transform};

    // Load accum
    if (tid < ${(mask_size + 1) * transform.threads_per_transform})
    {
        #pragma unroll
        for (unsigned int i = 0; i < ${p_ept}; i++)
        {
            shared_accum[mask_id * ${plength} + i * ${tpt} + thread_in_transform] =
                ${accum_a.load_combined_idx(slices)}(
                    batch_id, mask_id, i * ${tpt} + thread_in_transform);
        }
    }

    LOCAL_BARRIER;

    for (unsigned int bk_idx = 0; bk_idx < ${output_size}; bk_idx++)
    {

    ${bara.ctype} ai = ${bara.load_combined_idx(slices2)}(batch_id, bk_idx);

    {
        <%
            conversion_multiplier = plength // transform.transform_length;
        %>

        %for q in range(conversion_multiplier):
        int temp${q};
        %endfor

        #pragma unroll
        for (int i = tid; i < ${plength // conversion_multiplier}; i += ${local_size})
        {
            %for q in range(conversion_multiplier):
            int i${q} = i + ${transform.transform_length * q};
            unsigned int cmp${q} = (unsigned int)(i${q} < (ai & ${plength - 1}));
            unsigned int neg${q} = -(cmp${q} ^ (ai >> ${log2_plength}));
            unsigned int pos${q} = -((1 - cmp${q}) ^ (ai >> ${log2_plength}));
            %endfor

            %for mask_id in range(mask_size + 1):

                %for q in range(conversion_multiplier):
                temp${q} = shared_accum[(${mask_id * plength}) | ((i${q} - ai) & ${plength - 1})];
                temp${q} = (temp${q} & pos${q}) + ((-temp${q}) & neg${q});
                temp${q} -= shared_accum[(${mask_id * plength}) | i${q}];
                // decomp temp
                temp${q} += ${decomp_offset};
                %endfor

                %for decomp_id in range(decomp_length):
                    sh[${(decomp_id * (mask_size + 1) + mask_id) * sh_length_tr} + i] =
                        ${transform.module}i32_to_elem(
                            %for q in range(conversion_multiplier):
                            ((temp${q} >> ${32 - (decomp_id + 1) * bs_log2_base})
                                & ${decomp_mask}) - ${decomp_half}
                            %if q < conversion_multiplier - 1:
                            ,
                            %endif
                            %endfor
                        );
                %endfor
            %endfor
        }
    }

    LOCAL_BARRIER;

    if (tid < ${decomp_length * (mask_size + 1) * transform.threads_per_transform})
    {
        // Forward transform
        ${transform.module}forward_i32_shared(
            sh + (decomp_id * ${mask_size + 1} + mask_id) * ${sh_length_tr},
            (LOCAL_MEM_ARG ${transform.temp_ctype}*)(
                sh + (decomp_id * ${mask_size + 1} + mask_id) * ${sh_length_tr}),
            (${transform.module}CDATA_QUALIFIER ${transform.cdata_fw_ctype}*)${cdata_forward},
            thread_in_transform);
    }
    else
    {
        ${transform.module}noop_shared();
    }

    LOCAL_BARRIER;

    ## Iterating in reverse order, because the output shared array overlaps the input one.
    %for mask_out_id in reversed(range(mask_size + 1)):
    {
        ${tr_ctype} t, a, b;
        %for j in range(min_blocks(transform.transform_length, local_size)):
        {
            int idx = ${j * local_size} + tid;

            %if transform.transform_length % local_size != 0 and j == min_blocks(transform.transform_length, local_size) - 1:
            if (idx < ${transform.transform_length})
            {
            %endif

            t = ${tr_ctype}zero;
            %for mask_in_id in range(mask_size + 1):
            %for decomp_id in range(decomp_length):
            a = sh[${(decomp_id * (mask_size + 1) + mask_in_id) * sh_length_tr} + idx];
            b = ${tr_ctype}pack(
                    ${gsw.load_idx}(bk_idx, ${mask_in_id}, ${decomp_id}, ${mask_out_id}, idx));
            t = ${add}(t, ${mul_prepared}(a, b));
            %endfor
            %endfor

            <%
                temp_id = (
                    0 if mask_out_id == 0 else decomp_length * (mask_size + 1) - 1 + mask_out_id)
            %>
            sh[${temp_id * sh_length_tr} + idx] = t;

            %if transform.transform_length % local_size != 0 and j == min_blocks(transform.transform_length, local_size) - 1:
            }
            %endif
        }
        %endfor
    }
    LOCAL_BARRIER;
    %endfor

    // Inverse transform
    if (tid < ${(mask_size + 1) * transform.threads_per_transform})
    {
        // Following the temporary buffer usage scheme described at the beginning of the kernel.
        int temp_id = mask_id == 0 ? 0 : ${(mask_size + 1) * decomp_length - 1} + mask_id;

        ${transform.module}inverse_i32_shared_add(
            shared_accum + mask_id * ${plength},
            sh + temp_id * ${sh_length_tr},
            (LOCAL_MEM_ARG ${transform.temp_ctype}*)(sh + temp_id * ${sh_length_tr}),
            (${transform.module}CDATA_QUALIFIER ${transform.cdata_inv_ctype}*)${cdata_inverse},
            thread_in_transform);
    }
    else
    {
        ${transform.module}noop_shared();
    }

    LOCAL_BARRIER;
    }

    for (int i = tid; i <= ${input_size}; i += ${local_size})
    {
        if (i == ${input_size})
        {
            ${extracted_b.store_combined_idx(slices3)}(batch_id, shared_accum[${input_size}]);
        }
        else
        {
            ${extracted_a.store_combined_idx(slices2)}(
                batch_id, i, i == 0 ? shared_accum[0] : -shared_accum[${input_size} - i]);
        }
    }
}
</%def>
