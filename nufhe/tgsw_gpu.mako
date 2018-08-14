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
