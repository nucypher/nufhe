<%def name="encrypt_zero_fill_result(
    kernel_declaration, result_a, result_cv, noises1, noises2, ift_res)">

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    int batch_idx = virtual_global_id(0);
    int k_idx = virtual_global_id(1);
    int poly_idx = virtual_global_id(2);

    int result_a;

    if (k_idx < ${k})
    {
        result_a = ${noises1.load_combined_idx(noises1_slices)}(batch_idx, k_idx, poly_idx);
    }
    else
    {
        result_a = ${noises2.load_combined_idx(noises2_slices)}(batch_idx, poly_idx);
        for (int i = 0; i < ${k}; i++)
        {
            result_a += ${ift_res.load_combined_idx(noises1_slices)}(batch_idx, i, poly_idx);
        }
    }
    ${result_a.store_combined_idx(noises1_slices)}(batch_idx, k_idx, poly_idx, result_a);


    if (poly_idx == 0 && k_idx == 0)
    {
        ${result_cv.store_combined_idx(cv_slices)}(batch_idx, ${noise**2});
    }
}
</%def>
