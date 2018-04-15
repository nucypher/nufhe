<%def name="prepare_rfft_output(kernel_declaration, output, fft_results, A, B)">
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    VSIZE_T batch_id = virtual_global_id(0);
    VSIZE_T fft_id = virtual_global_id(1);

    ##G = numpy.empty(N//2 + 1, numpy.complex128)
    ##G[:N//2] = X * A + (numpy.roll(X[N//2-1::-1], 1)).conj() * B
    ##G[N//2] = X[0].real - X[0].imag

    int reverse_fft_id = fft_id == 0 ? 0 : ${N // 2} - fft_id;

    ${fft_results.ctype} X = ${fft_results.load_combined_idx(slices)}(batch_id, fft_id);
    ${fft_results.ctype} X_rev = ${fft_results.load_combined_idx(slices)}(batch_id, reverse_fft_id);

    %if not dont_store_last:
    if (fft_id == 0)
    {
        ${output.store_combined_idx(slices)}(
            batch_id, ${N // 2},
            COMPLEX_CTR(${output.ctype})(X.x - X.y, 0));
    }
    %endif

    ${A.ctype} A = ${A.load_idx}(fft_id);
    ${B.ctype} B = ${B.load_idx}(fft_id);

    ${output.store_combined_idx(slices)}(
        batch_id, fft_id,
        ${mul}(X, A) + ${mul}(${conj}(X_rev), B)
        );
}
</%def>
