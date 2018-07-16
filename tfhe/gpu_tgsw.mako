<%def name="TGswAddMuIntH(kernel_declaration, result_a, messages)">

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    int n_idx = virtual_global_id(0);

    int result_a;

    %for bloc in range(k+1):
    %for l_idx in range(l):
    result_a = ${result_a.load_idx}(n_idx, ${bloc}, ${l_idx}, ${bloc}, 0);
    ${result_a.store_idx}(
        n_idx, ${bloc}, ${l_idx}, ${bloc}, 0,
        result_a + ${messages.load_idx}(n_idx) * ${h[l_idx]});
    %endfor
    %endfor
}
</%def>
