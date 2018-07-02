<%def name="keyswitch(
    kernel_declaration, result_a, ks_a, ai)">

## inner_n
#define lwe_n 500

## outer_n
#define tlwe_n 1024
#define decomp_bits 2
#define decomp_size 8

## result_a: batch_shape + (inner_n,)
## ks_a: (inner_n, base, outer_n, t)
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
    int res = 0;
    int val = 0;

    for (int i = tid; i < lwe_n; i += bdim)
    {
        res = ${result_a.load_combined_idx(slices)}(batch_id, i);

        for (int j = 0; j < tlwe_n; j ++)
        {


            tmp = ${ai.load_combined_idx(slices)}(batch_id, j);
            tmp += decomp_offset;

            for (int k = 0; k < decomp_size; k++)
            {
                val = (tmp >> (32 - (k + 1) * decomp_bits)) & decomp_mask;
                if (val != 0)
                    res -= ${ks_a.load_idx}(j, k, val, i);
            }
        }
        ${result_a.store_combined_idx(slices)}(batch_id, i, res);
    }
}

</%def>
