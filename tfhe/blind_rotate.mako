<%def name="BlindRotate(
    kernel_declaration, accum_a, gsw, bara, cdata_forward, cdata_inverse, n)">
<%
    tpt = transform.threads_per_transform
    p_ept = transform.polynomial_length // transform.threads_per_transform
    tr_ept = transform.transform_length // transform.threads_per_transform

    # Makes the code a bit simpler, can be lifted if necessary
    assert transform.polynomial_length % tpt == 0
    assert transform.transform_length % tpt == 0
    assert transform.cdata_fw.size % tpt == 0
    assert transform.cdata_inv.size % tpt == 0
%>


${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    LOCAL_MEM ${transform.temp_ctype} transform_temp[${transform.temp_length}];
    LOCAL_MEM ${dtypes.ctype(transform.cdata_fw.dtype)} cdata_forward[${transform.cdata_fw.size}];
    LOCAL_MEM ${dtypes.ctype(transform.cdata_inv.dtype)} cdata_inverse[${transform.cdata_inv.size}];

    VSIZE_T batch_id = virtual_group_id(0);
    VSIZE_T tid = virtual_local_id(1);
    int thread_in_transform = tid;

    // Load constant data
    #pragma unroll
    for (int i = 0; i < ${transform.cdata_fw.size // tpt}; i++)
    {
        cdata_forward[i * ${tpt} + thread_in_transform] =
            ${cdata_forward.load_idx}(i * ${tpt} + thread_in_transform);
    }

    #pragma unroll
    for (int i = 0; i < ${transform.cdata_inv.size // tpt}; i++)
    {
        cdata_inverse[i * ${tpt} + thread_in_transform] =
            ${cdata_inverse.load_idx}(i * ${tpt} + thread_in_transform);
    }

    LOCAL_BARRIER; // Wait for the constant data to finish loading


    LOCAL_MEM ${accum_a.ctype} shared_accum[${transform.polynomial_length * (k + 1)}];

    ${accum_a.ctype} accum_a[${p_ept * (k + 1)}];
    ${tr_ctype} tmpa_a[${tr_ept * (k + 1)}];
    ${tr_ctype} decaFFT[${tr_ept * (k + 1) * l}];

    // Load accum
    %for k_in_id in range(k + 1):
        for (int i = 0; i < ${p_ept}; i++)
        {
            shared_accum[${k_in_id * transform.polynomial_length} + i * ${tpt} + tid] =
                ${accum_a.load_combined_idx(slices)}(batch_id, ${k_in_id}, i * ${tpt} + tid);
        }
    %endfor

    LOCAL_BARRIER;


    ${accum_a.ctype} tr_accum_a[${p_ept}];

    for (int bk_idx = 0; bk_idx < ${n}; bk_idx++)
    {

    LOCAL_MEM_ARG ${accum_a.ctype}* current_accum;
    ${bara.ctype} ai = ${bara.load_idx}(batch_id, bk_idx);

    %for k_in_id in range(k + 1):

        //
        current_accum = shared_accum + ${k_in_id * transform.polynomial_length};

        for (int i = 0; i < ${p_ept}; i++)
        {
            int elem_id = i * ${tpt} + tid;

            ${accum_a.ctype} res = 0;
            if (ai < ${transform.polynomial_length})
            {
                if (elem_id < ai)
                {
                    res = -current_accum[elem_id + ${transform.polynomial_length} - ai];
                }
                else
                {
                    res = current_accum[elem_id - ai];
                }
            }
            else
            {
                ${bara.ctype} aa = ai - ${transform.polynomial_length};
                if (elem_id < aa)
                {
                    res = current_accum[elem_id + ${transform.polynomial_length} - aa];
                }
                else
                {
                    res = -current_accum[elem_id - aa];
                }
            }
            res -= current_accum[elem_id];

            accum_a[i + ${k_in_id * p_ept}] = res;
        }
    %endfor

    LOCAL_BARRIER;

    %for k_in_id in range(k + 1):
    %for l_id in range(l):
        // TGswTorus32PolynomialDecompH
        for (int i = 0; i < ${p_ept}; i++)
        {
            ${accum_a.ctype} sample = accum_a[i + ${k_in_id * p_ept}];
            int p = ${l_id} + 1;
            int decal = 32 - p * ${params.Bgbit};
            tr_accum_a[i] = (
                (((sample + (${params.offset})) >> decal) & ${params.maskMod}) - ${params.halfBg}
            );
        }
        // Forward transform
        ${transform.module}forward_i32(
            decaFFT + ${tr_ept * l * k_in_id + tr_ept * l_id},
            tr_accum_a,
            transform_temp,
            (${transform.cdata_fw_ctype}*)cdata_forward,
            thread_in_transform);
        LOCAL_BARRIER;
    %endfor
    %endfor

    // TLweFFTAddMulRTo
    for (int i = 0; i < ${tr_ept * (k + 1)}; i++)
    {
        tmpa_a[i] = ${tr_ctype}zero;
    }

    %for k_out_id in range(k + 1):
    %for k_in_id in range(k + 1):
    %for l_id in range(l):
        for (int i = 0; i < ${tr_ept}; i++)
        {
            ${tr_ctype} a = decaFFT[i + ${tr_ept * l * k_in_id + tr_ept * l_id}];
            ${tr_ctype} b = ${tr_ctype}pack(
                ${gsw.load_idx}(
                    bk_idx, ${k_in_id}, ${l_id}, ${k_out_id}, i * ${tpt} + tid)
                );
            tmpa_a[i + ${k_out_id * tr_ept}] = ${add}(tmpa_a[i + ${k_out_id * tr_ept}], ${mul}(a, b));
        }
    %endfor
    %endfor
    %endfor

    // Inverse transform
    %for k_out_id in range(k + 1):
        ${transform.module}inverse_i32(
            accum_a + ${p_ept * k_out_id},
            tmpa_a + ${tr_ept * k_out_id},
            transform_temp,
            (${transform.cdata_inv_ctype}*)cdata_inverse,
            thread_in_transform);
        LOCAL_BARRIER;

        #pragma unroll
        for (int i = 0; i < ${p_ept}; i++)
        {
            shared_accum[${k_out_id * transform.polynomial_length} + i * ${tpt} + tid] +=
                accum_a[${k_out_id * p_ept} + i];
        }
        LOCAL_BARRIER;

    %endfor
    LOCAL_BARRIER;
    }

    %for k_out_id in range(k + 1):
    #pragma unroll
    for (int i = 0; i < ${p_ept}; i++)
        ${accum_a.store_combined_idx(slices)}(
            batch_id, ${k_out_id}, thread_in_transform + i * ${tpt},
            shared_accum[${k_out_id * transform.polynomial_length} + i * ${tpt} + tid]);
    %endfor
}
</%def>
