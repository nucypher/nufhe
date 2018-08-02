import numpy


def int_prod(arr):
    return numpy.prod(arr, dtype=numpy.int32)


# minus_one=True: result = (X^ai-1) * source
# minus_one=False: result = X^{a} * source
def ShiftTorusPolynomial_ref(ais, arr, ai_view=False, minus_one=False, invert_ais=False):

    if ai_view:
        assert len(ais.shape) == len(arr.shape) - 1
        assert ais.shape[:-1] == arr.shape[:-2]
    else:
        assert len(ais.shape) == len(arr.shape) - 1
        assert ais.shape == arr.shape[:-1]

    N = arr.shape[-1]

    def _kernel(output, ais, ai_idx, input_):

        if ai_view:
            ai_batch_len = int_prod(ais.shape[:-int(ai_view)])
            arr_batch_len = int_prod(arr.shape[len(ais.shape)-1:-1])

            ais = ais.reshape(ai_batch_len, ais.shape[-1])
            ais = ais[:,ai_idx]

            out_c = output.reshape(ai_batch_len, arr_batch_len, N)
            in_c = input_.reshape(ai_batch_len, arr_batch_len, N)

        else:
            ais = ais.flatten()
            ai_batch_len = ais.size

            out_c = output.reshape(ai_batch_len, 1, N)
            in_c = input_.reshape(ai_batch_len, 1, N)


        if invert_ais:
            ais = 2 * N - ais

        for i in range(ai_batch_len):
            ai = ais[i]
            if ai < N:
                out_c[i,:,:ai] = -in_c[i,:,(N-ai):N]
                out_c[i,:,ai:N] = in_c[i,:,:(N-ai)]
            else:
                aa = ai - N
                out_c[i,:,:aa] = in_c[i,:,(N-aa):N]
                out_c[i,:,aa:N] = -in_c[i,:,:(N-aa)]

        if minus_one:
            out_c -= in_c

    return _kernel
