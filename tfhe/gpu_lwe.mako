<%def name="build_keyswitch_key(
        idxs, ks_a, ks_b, ks_cv, in_key, b_term, noises_a, noises_b, noises_b_mean)">

    int extracted_n_idx = ${idxs[0]};
    int t_idx = ${idxs[1]};
    int base_idx = ${idxs[2]};

    if (base_idx == 0)
    {
        for (int i = 0; i < ${inner_n}; i++)
        {
            ${ks_a.store_idx}(extracted_n_idx, t_idx, 0, i, 0);
        }
        ${ks_b.store_idx}(extracted_n_idx, t_idx, 0, 0);
        ${ks_cv.store_idx}(extracted_n_idx, t_idx, 0, 0);
    }
    else
    {
        int h = base_idx + 1;
        int j = t_idx + 1;
        int key = ${in_key.load_idx}(extracted_n_idx);

        int message = key * (h - 1) * (1 << (32 - j * ${basebit}));

        ${noises_b.ctype} noise_b =
            ${noises_b.load_idx}(extracted_n_idx, t_idx, base_idx - 1)
            - ${noises_b_mean.load_idx}();

        ${ks_cv.store_idx}(extracted_n_idx, t_idx, base_idx, ${alpha**2});

        for (int i = 0; i < ${inner_n}; i++)
        {
            ${ks_a.store_idx}(
                extracted_n_idx, t_idx, base_idx, i,
                ${noises_a.load_idx}(extracted_n_idx, t_idx, base_idx - 1, i));
        }

        ${ks_b.store_idx}(
            extracted_n_idx, t_idx, base_idx,
            message + ${dtot32}(noise_b) + ${b_term.load_idx}(extracted_n_idx, t_idx, base_idx - 1)
            );
    }
</%def>


<%def name="keyswitch(
    kernel_declaration, result_a, result_b, result_cv, ks_a, ks_b, ks_cv, ai)">

## inner_n
#define lwe_n 500

## outer_n
#define tlwe_n 1024

#define decomp_bits 2
#define decomp_size 8

## result_a: batch_shape + (inner_n,)
## ks_a: (outer_n, t, base, inner_n)
## ai: batch_shape + (outer_n,)

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    const int decomp_mask = (1u << decomp_bits) - 1;
    const int decomp_offset = 1u << (31 - decomp_size * decomp_bits);
    unsigned int tid = virtual_local_id(1);
    unsigned int bdim = virtual_local_size(1);
    unsigned int batch_id = virtual_global_id(0);

    int tmp;
    int res_a = 0;
    int res_b = 0;
    double res_cv = 0;
    int val = 0;

    for (int i = tid; i < lwe_n; i += bdim)
    {
        res_a = ${result_a.load_combined_idx(slices)}(batch_id, i);

        if (i == 0)
        {
            res_b = ${result_b.load_combined_idx(slices[:-1])}(batch_id);
            res_cv = ${result_cv.load_combined_idx(slices[:-1])}(batch_id);
        }

        for (int j = 0; j < tlwe_n; j ++)
        {
            tmp = ${ai.load_combined_idx(slices)}(batch_id, j);
            tmp += decomp_offset;

            for (int k = 0; k < decomp_size; k++)
            {
                val = (tmp >> (32 - (k + 1) * decomp_bits)) & decomp_mask;
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
