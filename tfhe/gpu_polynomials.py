import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.fft import FFT
from reikna.algorithms import PureParallel
import reikna.transformations as transformations
from reikna.cluda import dtypes, functions

from .polynomials import TorusPolynomialArray, FFT_COEFF
from .computation_cache import get_computation
from .numeric_functions import Torus32, Float
from .fft_specialized import RFFT, IRFFT, RTFFT


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


def transform_i2c_v2_input(arr, output_dtype, coeff):
    # input: int, ... x N
    # output: float, ... x N

    result_arr = Type(output_dtype, arr.shape)

    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        ${output.store_same}((${output.ctype})(${input.load_same}) / ${coeff});
        """,
        render_kwds=dict(coeff=coeff),
        connectors=['output', 'input'])


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


def transform_i2c_output_v3(input_):
    # input: complex, ... x (N + 1)
    # output: complex, ... x N//2

    N = input_.shape[-1] - 1
    result_arr = Type(input_.dtype, input_.shape[:-1] + (N // 2,))

    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(input_, 'i')),
        ],
        """
        if (${idxs[-1]} % 2 == 1)
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
            coeff=FFT_COEFF, i64_ctype=dtypes.ctype(numpy.int64)),
        connectors=['input'])


class I2C_FFT(Computation):

    def __init__(self, arr, coeff):
        # coeff=2 to replicate ip_ifft
        # coeff=2^33 to replicate tp_ifft

        output_r_dtype = Float
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


class I2C_FFT_v2(Computation):
    """
    I2C FFT using a real-to-complex FFT
    that takes into account the translational symmetry we have.
    """

    def __init__(self, arr, coeff):
        # coeff=2 to replicate ip_ifft
        # coeff=2^33 to replicate tp_ifft

        N = arr.shape[-1]

        fft_arr = Type(Float, arr.shape)
        tr_input = transform_i2c_v2_input(arr, Float, coeff)

        rtfft = RTFFT(fft_arr)
        rtfft.parameter.input.connect(
            tr_input, tr_input.output, input_poly=tr_input.input)

        self._rtfft = rtfft

        Computation.__init__(self, [
            Parameter('output', Annotation(rtfft.parameter.output, 'o')),
            Parameter('input', Annotation(arr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()
        plan.computation_call(self._rtfft, output, input_)
        return plan


class I2C_FFT_v3(Computation):
    """
    I2C FFT using a real-to-complex FFT.
    """

    def __init__(self, arr, coeff):
        # coeff=2 to replicate ip_ifft
        # coeff=2^33 to replicate tp_ifft

        output_r_dtype = Float
        N = arr.shape[-1]

        fft_arr = Type(output_r_dtype, arr.shape[:-1] + (2*N,))
        fft = RFFT(fft_arr)

        tr_input = transform_i2c_input(arr, output_r_dtype, coeff)
        tr_output = transform_i2c_output_v3(fft.parameter.output)

        fft.parameter.input.connect(
            tr_input, tr_input.output, input_poly=tr_input.input)
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

        output_dtype = Torus32

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



def transform_c2i_input_v2(arr):
    # input: complex, ... x N//2
    # output: complex, ... x N+1

    N = arr.shape[-1] * 2
    result_arr = Type(arr.dtype, arr.shape[:-1] + (N + 1,))

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
        else
        {
            ${output.store_same}(
                ${input.load_idx}(${", ".join(idxs[:-1])}, (${idxs[-1]} - 1) / 2)
            );
        }
        """,
        render_kwds=dict(N=N, ctype=dtypes.ctype(arr.dtype), conj=functions.conj(arr.dtype)),
        connectors=['output'])


def transform_c2i_output_v2(arr, output_dtype):
    # input: real, ... x 2N
    # output: Torus, ... x N

    N = arr.shape[-1] // 2

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
            coeff=FFT_COEFF, i64_ctype=dtypes.ctype(numpy.int64)),
        connectors=['input'])


class C2I_FFT_v2(Computation):

    def __init__(self, arr):

        output_dtype = Torus32

        N = arr.shape[-1] * 2

        fft_arr = Type(arr.dtype, arr.shape[:-1] + (N + 1,))

        fft = IRFFT(fft_arr)

        tr_input = transform_c2i_input_v2(arr)
        tr_output = transform_c2i_output_v2(fft.parameter.output, output_dtype)

        fft.parameter.input.connect(tr_input, tr_input.output, input_poly=tr_input.input)
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


def transform_mul_by_xai(ais, arr, ai_view=False, minus_one=False, invert_ais=False):
    # arr: ... x N
    # ais: arr.shape[0], int

    assert len(ais.shape) == 2 if ai_view else 1
    assert ais.shape[0] == arr.shape[0]
    N = arr.shape[-1]

    return Transformation(
        [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('ais', Annotation(ais, 'i')),
            Parameter('ai_idx', Annotation(numpy.int32)), # FIXME: unused if ai_view==False
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        %if ai_view:
        ${ai_ctype} ai = ${ais.load_idx}(${idxs[0]}, ${ai_idx});
        %else:
        ${ai_ctype} ai = ${ais.load_idx}(${idxs[0]});
        %endif

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
            N=N, ai_ctype=dtypes.ctype(ais.dtype),
            ai_view=ai_view, minus_one=minus_one, invert_ais=invert_ais),
        connectors=['output'])


class TPMulByXai(Computation):

    def __init__(self, ais, arr, ai_view=False, minus_one=False, invert_ais=False):
        # `invert_ais` means that `2N - ais` will be used instead of `ais`
        tr = transform_mul_by_xai(
            ais, arr, ai_view=ai_view, minus_one=minus_one, invert_ais=invert_ais)
        self._pp = PureParallel.from_trf(tr, guiding_array=tr.output)

        Computation.__init__(self, [
            Parameter('output', Annotation(self._pp.parameter.output, 'o')),
            Parameter('ais', Annotation(self._pp.parameter.ais, 'i')),
            Parameter('ai_idx', Annotation(numpy.int32)), # FIXME: unused if ai_view==False
            Parameter('input', Annotation(self._pp.parameter.input, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, ais, ai_idx, input_):
        plan = plan_factory()
        plan.computation_call(self._pp, output, ais, ai_idx, input_)
        return plan


# result= X^{a}*source
def tp_mul_by_xai_gpu(out: TorusPolynomialArray, ais, in_: TorusPolynomialArray, invert_ais=False):
    thr = out.coefsT.thread
    comp = get_computation(
        thr, TPMulByXai, ais, in_.coefsT, minus_one=False, invert_ais=invert_ais)
    comp(out.coefsT, ais, 0, in_.coefsT)


# result = (X^ai-1) * source
def tp_mul_by_xai_minus_one_gpu(out: TorusPolynomialArray, ais, ai_idx, in_: TorusPolynomialArray):
    thr = out.coefsT.thread
    comp = get_computation(
        thr, TPMulByXai, ais, in_.coefsT, ai_view=True, minus_one=True, invert_ais=False)
    comp(out.coefsT, ais, ai_idx, in_.coefsT)
