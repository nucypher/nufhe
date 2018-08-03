<%def name="shift_torus_polynomial(kernel_declaration, result, source, powers, powers_idx)">
<%
    poly_slices = (batch_len, poly_batch_len, 1)
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    int batch_id = virtual_global_id(0);
    int poly_batch_id = virtual_global_id(1);
    int polynomial_id = virtual_global_id(2);

    %if powers_view:
    ${powers.ctype} power = ${powers.load_combined_idx((batch_len, 1))}(batch_id, ${powers_idx});
    %else:
    ${powers.ctype} power = ${powers.load_combined_idx((batch_len,))}(batch_id);
    %endif

    %if invert_powers:
    power = ${2 * polynomial_degree} - power;
    %endif

    ${result.ctype} res;

    if (power < ${polynomial_degree})
    {
        if (polynomial_id < power)
        {
            res = -${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id + ${polynomial_degree} - power);
        }
        else
        {
            res = ${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id - power);
        }
    }
    else
    {
        power = power - ${polynomial_degree};
        if (polynomial_id < power)
        {
            res = ${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id + ${polynomial_degree} - power);
        }
        else
        {
            res = -${source.load_combined_idx(poly_slices)}(
                batch_id, poly_batch_id, polynomial_id - power);
        }
    }

    %if minus_one:
    res -= ${source.load_combined_idx(poly_slices)}(batch_id, poly_batch_id, polynomial_id);
    %endif

    ${result.store_combined_idx(poly_slices)}(batch_id, poly_batch_id, polynomial_id, res);
}
</%def>
