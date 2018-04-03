import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.fft import FFT
from reikna.algorithms import PureParallel
import reikna.transformations as transformations
from reikna.cluda import dtypes, functions


# p: Int (coeff=2)/Torus (coeff=2^33)
# result: Complex
def ip_ifft_reference(p, coeff):
    a = p.reshape(numpy.prod(p.shape[:-1]), p.shape[-1])
    N = a.shape[-1]

    res = numpy.empty((a.shape[0], N//2), numpy.complex128)

    in_arr = numpy.empty((a.shape[0], 2 * N), numpy.float64)

    in_arr[:,:N] = a / coeff
    in_arr[:,N:] = -in_arr[:,:N]

    out_arr = numpy.fft.rfft(in_arr)

    res[:,:N//2] = out_arr[:,1:N+1:2]

    return res.reshape(p.shape[:-1] + (N//2,))


# p: Complex
# result: Torus
def tp_fft_reference(p):
    a = p.reshape(numpy.prod(p.shape[:-1]), p.shape[-1])
    N = a.shape[-1] * 2

    res = numpy.empty((a.shape[0], N), numpy.int32)

    in_arr = numpy.empty((res.shape[0], N + 1), numpy.complex128)
    in_arr[:,0:N+1:2] = 0
    in_arr[:,1:N+1:2] = a

    out_arr = numpy.fft.irfft(in_arr)

    # the first part is from the original libtfhe;
    # the second part is from a different FFT scaling in Julia
    coeff = (2**32 / N) * (2 * N)

    res[:,:] = numpy.round(out_arr[:,:N] * coeff).astype(numpy.int64).astype(numpy.int32)

    return res.reshape(p.shape[:-1] + (N,))


def transform_i2c_input(arr, output_dtype, coeff):
    # input: int, ... x N
    # output: float, ... x 2N

    N = arr.shape[-1]
    result_arr = Type(output_dtype, arr.shape[:-1] + (2 * N,))

    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        if (${idxs[-1]} < ${N})
        {
            ${output.store_same}(
                (${out_ctype})(${input.load_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]})) / ${coeff}
            );
        }
        else
        {
            ${output.store_same}(
                -(${out_ctype})(${input.load_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]} - ${N})) / ${coeff}
            );
        }
        """,
        render_kwds=dict(N=N, coeff=coeff, out_ctype=dtypes.ctype(output_dtype)),
        connectors=['output'])


def transform_i2c_output(arr):
    # input: complex, ... x 2N
    # output: complex, ... x N//2

    N = arr.shape[-1] // 2
    result_arr = Type(arr.dtype, arr.shape[:-1] + (N // 2,))

    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        if (${idxs[-1]} < ${N + 1} && ${idxs[-1]} % 2 == 1)
        {
            ${output.store_idx}(
                ${", ".join(idxs[:-1])}, (${idxs[-1]} - 1) / 2,
                ${input.load_same}
            );
        }
        """,
        render_kwds=dict(N=N),
        connectors=['input'])



def transform_c2i_input(arr):
    # input: complex, ... x N//2
    # output: complex, ... x 2*N

    N = arr.shape[-1] * 2
    result_arr = Type(arr.dtype, arr.shape[:-1] + (2 * N,))

    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        if (${idxs[-1]} % 2 == 0)
        {
            ${output.store_same}(COMPLEX_CTR(${ctype})(0, 0));
        }
        else if(${idxs[-1]} < ${N})
        {
            ${output.store_same}(
                ${input.load_idx}(${", ".join(idxs[:-1])}, (${idxs[-1]} - 1) / 2)
            );
        }
        else
        {
            ${output.store_same}(
                ${conj}(${input.load_idx}(${", ".join(idxs[:-1])}, ((${2 * N} - ${idxs[-1]}) - 1) / 2))
            );
        }
        """,
        render_kwds=dict(N=N, ctype=dtypes.ctype(arr.dtype), conj=functions.conj(arr.dtype)),
        connectors=['output'])


def transform_c2i_output(arr, output_dtype):
    # input: complex, ... x 2N
    # output: Torus, ... x N

    N = arr.shape[-1] // 2

    assert (2**32) % N == 0
    coeff = (2**32 // N) * (2 * N)

    result_arr = Type(output_dtype, arr.shape[:-1] + (N,))

    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        if (${idxs[-1]} < ${N})
        {
            ${output.store_same}(
                (${out_ctype})((${i64_ctype})(round(${input.load_same} * ${coeff})))
            );
        }
        """,
        render_kwds=dict(
            N=N, out_ctype=dtypes.ctype(output_dtype),
            coeff=coeff, i64_ctype=dtypes.ctype(numpy.int64)),
        connectors=['input'])


class I2C_FFT(Computation):

    def __init__(self, arr, coeff):
        # coeff=2 to replicate ip_ifft
        # coeff=2^33 to replicate tp_ifft

        output_r_dtype = numpy.float64
        output_c_dtype = dtypes.complex_for(output_r_dtype)
        N = arr.shape[-1]

        fft_arr = Type(output_c_dtype, arr.shape[:-1] + (2*N,))

        tr_input = transform_i2c_input(arr, output_r_dtype, coeff)
        zero_imag = transformations.broadcast_const(tr_input.output, 0)
        make_complex = transformations.combine_complex(fft_arr)
        tr_output = transform_i2c_output(fft_arr)

        fft = FFT(fft_arr, axes=(len(arr.shape)-1,))
        fft.parameter.input.connect(
            make_complex, make_complex.output,
            input_real=make_complex.real, input_imag=make_complex.imag)
        fft.parameter.input_real.connect(
            tr_input, tr_input.output, input_poly=tr_input.input)
        fft.parameter.input_imag.connect(
            zero_imag, zero_imag.output)
        fft.parameter.output.connect(
            tr_output, tr_output.input, output_poly=tr_output.output)

        self._fft = fft

        Computation.__init__(self, [
            Parameter('output', Annotation(self._fft.parameter.output_poly, 'o')),
            Parameter('input', Annotation(self._fft.parameter.input_poly, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()
        plan.computation_call(self._fft, output, input_)
        return plan


class C2I_FFT(Computation):

    def __init__(self, arr):

        output_dtype = numpy.int32 # FIXME: technically, should be Torus32

        N = arr.shape[-1] * 2

        fft_arr = Type(arr.dtype, arr.shape[:-1] + (2*N,))

        tr_input = transform_c2i_input(arr)
        split_complex = transformations.split_complex(fft_arr)
        ignore_imag = transformations.ignore(split_complex.imag)
        tr_output = transform_c2i_output(split_complex.real, output_dtype)

        fft = FFT(fft_arr, axes=(len(arr.shape)-1,))
        fft.parameter.input.connect(tr_input, tr_input.output, input_poly=tr_input.input)
        fft.parameter.output.connect(
            split_complex, split_complex.input,
            output_real=split_complex.real, output_imag=split_complex.imag)
        fft.parameter.output_imag.connect(ignore_imag, ignore_imag.input)
        fft.parameter.output_real.connect(
            tr_output, tr_output.input, output_poly=tr_output.output)

        self._fft = fft

        Computation.__init__(self, [
            Parameter('output', Annotation(self._fft.parameter.output_poly, 'o')),
            Parameter('input', Annotation(self._fft.parameter.input_poly, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()
        plan.computation_call(self._fft, output, input_, inverse=True)
        return plan


# result = (X^ai-1) * source
def tp_mul_by_xai_minus_one_(out, ais, in_):
    out_c = out #.coefsT
    in_c = in_ #.coefsT

    N = out_c.shape[-1]
    for i in range(out.shape[0]):
        ai = ais[i]
        if ai < N:
            out_c[i,:,:ai] = -in_c[i,:,(N-ai):N] - in_c[i,:,:ai] # sur que i-a<0
            out_c[i,:,ai:N] = in_c[i,:,:(N-ai)] - in_c[i,:,ai:N] # sur que N>i-a>=0
        else:
            aa = ai - N
            out_c[i,:,:aa] = in_c[i,:,(N-aa):N] - in_c[i,:,:aa] # sur que i-a<0
            out_c[i,:,aa:N] = -in_c[i,:,:(N-aa)] - in_c[i,:,aa:N] # sur que N>i-a>=0


# result= X^{a}*source
def tp_mul_by_xai_(out, ais, in_):
    out_c = out #.coefsT
    in_c = in_ #.coefsT

    N = out_c.shape[-1]
    for i in range(out.shape[0]):
        ai = ais[i]
        if ai < N:
            out_c[i,:ai] = -in_c[i,(N-ai):N] # sur que i-a<0
            out_c[i,ai:N] = in_c[i,:(N-ai)] # sur que N>i-a>=0
        else:
            aa = ai - N
            out_c[i,:aa] = in_c[i,(N-aa):N] # sur que i-a<0
            out_c[i,aa:N] = -in_c[i,:(N-aa)] # sur que N>i-a>=0


def transform_mul_by_xai(ais, arr, minus_one=False, invert_ais=False):
    # arr: ... x N
    # ais: arr.shape[0], int

    assert len(ais.shape) == 1 and ais.shape[0] == arr.shape[0]
    N = arr.shape[-1]

    return Transformation(
        [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('ais', Annotation(ais, 'i')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        ${ai_ctype} ai = ${ais.load_idx}(${idxs[0]});

        %if invert_ais:
        ai = ${2 * N} - ai;
        %endif

        ${output.ctype} res;

        if (ai < ${N})
        {
            if (${idxs[-1]} < ai)
            {
                res = -${input.load_idx}(
                        ${", ".join(idxs[:-1])}, ${idxs[-1]} + ${N} - ai
                        );
            }
            else
            {
                res = ${input.load_idx}(
                        ${", ".join(idxs[:-1])}, ${idxs[-1]} - ai
                        );
            }
        }
        else
        {
            ${ai_ctype} aa = ai - ${N};
            if (${idxs[-1]} < aa)
            {
                res = ${input.load_idx}(
                        ${", ".join(idxs[:-1])}, ${idxs[-1]} + ${N} - aa
                        );
            }
            else
            {
                res = -${input.load_idx}(
                      ${", ".join(idxs[:-1])}, ${idxs[-1]} - aa
                      );
            }
        }

        %if minus_one:
        res -= ${input.load_same};
        %endif

        ${output.store_same}(res);
        """,
        render_kwds=dict(
            N=N, ai_ctype=dtypes.ctype(ais.dtype), minus_one=minus_one, invert_ais=invert_ais),
        connectors=['output'])


class TPMulByXai(Computation):

    def __init__(self, ais, arr, minus_one=False, invert_ais=False):
        # `invert_ais` means that `2N - ais` will be used instead of `ais`

        tr = transform_mul_by_xai(ais, arr, minus_one=minus_one, invert_ais=invert_ais)
        self._pp = PureParallel.from_trf(tr, guiding_array=tr.output)

        Computation.__init__(self, [
            Parameter('output', Annotation(self._pp.parameter.output, 'o')),
            Parameter('ais', Annotation(self._pp.parameter.ais, 'i')),
            Parameter('input', Annotation(self._pp.parameter.input, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, ais, input_):
        plan = plan_factory()
        plan.computation_call(self._pp, output, ais, input_)
        return plan


def test_mul_by_xai():

    from reikna.cluda import ocl_api

    numpy.random.seed(125)

    N = 16

    api = ocl_api()
    thr = api.Thread.create()


    data = numpy.random.randint(0, 10000, size=(300, N))
    ais = numpy.random.randint(0, 2 * N, size=300)
    data_dev = thr.to_device(data)
    ais_dev = thr.to_device(ais)

    comp = TPMulByXai(ais, data, minus_one=False).compile(thr)
    res_dev = thr.empty_like(comp.parameter.output)

    comp(res_dev, ais_dev, data_dev)
    res_reikna = res_dev.get()

    res_ref = numpy.empty_like(data)
    tp_mul_by_xai_(res_ref, ais, data)

    assert numpy.allclose(res_reikna, res_ref)


    data = numpy.random.randint(0, 10000, size=(300, 10, N))
    ais = numpy.random.randint(0, 2 * N, size=300)
    data_dev = thr.to_device(data)
    ais_dev = thr.to_device(ais)

    comp = TPMulByXai(ais, data, minus_one=True).compile(thr)
    res_dev = thr.empty_like(comp.parameter.output)

    comp(res_dev, ais_dev, data_dev)
    res_reikna = res_dev.get()

    res_ref = numpy.empty_like(data)
    tp_mul_by_xai_minus_one_(res_ref, ais, data)

    assert numpy.allclose(res_reikna, res_ref)



def test_fft():
    from reikna.cluda import ocl_api

    numpy.random.seed(125)
    N = 1024

    api = ocl_api()
    thr = api.Thread.create()

    data = numpy.random.randint(-2**31, 2**31, size=(500, 2, 2, N), dtype=numpy.int32)

    ipfft = I2C_FFT(data, 2).compile(thr)

    data_dev = thr.to_device(data)
    res_dev = thr.empty_like(ipfft.parameter.output)

    ipfft(res_dev, data_dev)
    res_reikna = res_dev.get()
    res_ref = ip_ifft_reference(data, 2)

    assert numpy.allclose(res_reikna, res_ref)


    #size = (500, 2, 2, N//2)
    size = (2, 1024)
    data = (
        numpy.random.normal(size=size)
        + 1j * numpy.random.normal(size=size))

    pfft = C2I_FFT(data).compile(thr)

    data_dev = thr.to_device(data)
    res_dev = thr.empty_like(pfft.parameter.output)

    pfft(res_dev, data_dev)
    res_reikna = res_dev.get()
    res_ref = tp_fft_reference(data)

    print(res_reikna)
    print(res_ref)

    assert numpy.allclose(res_reikna, res_ref)


if __name__ == '__main__':

    test_mul_by_xai()

