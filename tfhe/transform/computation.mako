<%def name="standalone_transform(kernel_declaration, output, input_, cdata)">
<%
    tpt = transform.threads_per_transform

    tr_cdata = transform.cdata_inv if inverse else transform.cdata_fw
    tr_cdata_ctype = transform.cdata_inv_ctype if inverse else transform.cdata_fw_ctype

    # Makes the code a bit simpler, can be lifted if necessary
    assert transform.polynomial_length % tpt == 0
    assert transform.transform_length % tpt == 0
    assert tr_cdata.size % tpt == 0

    params_poly = dict(
        ept=transform.polynomial_length // tpt,
        dtype=transform.polynomial_dtype,
        ctype=transform.polynomial_ctype)
    params_tr = dict(
        ept=transform.transform_length // tpt,
        dtype=transform.elem_dtype,
        ctype=transform.elem_ctype)

    if i32_conversion:
        if inverse:
            params_in = params_tr
            params_out = params_poly
        else:
            params_in = params_poly
            params_out = params_tr
    else:
        params_in = params_tr
        params_out = params_tr

    tpb = transforms_per_block
%>


${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    LOCAL_MEM ${dtypes.ctype(transform.temp_dtype)} temp[${transform.temp_length} * ${tpb}];
    LOCAL_MEM ${dtypes.ctype(tr_cdata.dtype)} cdata[${tr_cdata.size}];

    ${dtypes.ctype(params_in['dtype'])} r_in[${params_in['ept']}];
    ${dtypes.ctype(params_out['dtype'])} r_out[${params_out['ept']}];

    VSIZE_T batch_id = virtual_group_id(0);
    VSIZE_T tid = virtual_local_id(1);
    int transform_in_block = tid / ${tpt};
    int thread_in_transform = tid % ${tpt};

    // Load constant data
    if (transform_in_block == 0)
    {
        #pragma unroll
        for (int i = 0; i < ${tr_cdata.size // tpt}; i++)
        {
            cdata[i * ${tpt} + thread_in_transform] =
                ${cdata.load_idx}(i * ${tpt} + thread_in_transform);
        }
    }

    %if batch_size % tpb != 0:
    bool active_thread = (batch_id < ${blocks_num - 1} || transform_in_block < ${batch_size % tpb});
    %endif

    %if batch_size % tpb != 0:
    if (active_thread)
    {
    %endif
    #pragma unroll
    for (int i = 0; i < ${params_in['ept']}; i++)
    {
        r_in[i] = ${input_.load_combined_idx(slices)}(
            batch_id * ${tpb} + transform_in_block, thread_in_transform + i * ${tpt});
    }
    %if batch_size % tpb != 0:
    }
    %endif

    LOCAL_BARRIER; // Wait for the constant data to finish loading

    %if kernel_repetitions > 1:
    for (int i = 0; i < ${kernel_repetitions}; i++)
    {
    %endif

    %if batch_size % tpb != 0:
    if (active_thread)
    {
    %endif
        LOCAL_MEM_ARG ${dtypes.ctype(transform.temp_dtype)}* transform_temp =
            temp + ${transform.temp_length} * transform_in_block;
        %if i32_conversion:
            %if inverse:
                ${transform.module}inverse_i32(
            %else:
                ${transform.module}forward_i32(
            %endif
        %else:
            %if inverse:
                ${transform.module}inverse(
            %else:
                ${transform.module}forward(
            %endif
        %endif
            (${params_out['ctype']}*)r_out,
            (${params_in['ctype']}*)r_in,
            (LOCAL_MEM_ARG ${transform.temp_ctype}*)transform_temp,
            (LOCAL_MEM_ARG ${tr_cdata_ctype}*)cdata,
            thread_in_transform);
    %if batch_size % tpb != 0:
    }
    else
    {
        ${transform.module}noop();
    }
    %endif

    %if kernel_repetitions > 1:
    }
    %endif

    %if batch_size % tpb != 0:
    if (active_thread)
    {
    %endif
    #pragma unroll
    for (int i = 0; i < ${params_out['ept']}; i++)
    {
        ${output.store_combined_idx(slices)}(
            batch_id * ${tpb} + transform_in_block, thread_in_transform + i * ${tpt},
            r_out[i]);
    }
    %if batch_size % tpb != 0:
    }
    %endif
}

</%def>
