import numpy

from reikna.fft import FFT

import reikna.helpers as helpers
from reikna.cluda import dtypes
from reikna.core import Computation, Parameter, Annotation, Type, Transformation
import reikna.cluda.functions as functions
from reikna.algorithms import Reduce, Scan, predicate_sum

TEMPLATE = helpers.template_for(__file__)


"""
Source of the algorithms:
L. R. Rabiner, "On the Use of Symmetry in FFT Computation"
IEEE Transactions on Acoustics, Speech, and Signal Processing 27(3), 233-239 (1979)
doi: 10.1109/TASSP.1979.1163235
"""


def my_rfft(a):

    N = a.size

    WNmk = numpy.exp(-2j * numpy.pi * numpy.arange(N//2) / N)
    A = 0.5 * (1 - 1j * WNmk)
    B = 0.5 * (1 + 1j * WNmk)

    x = a[::2] + 1j * a[1::2]
    X = numpy.fft.fft(x)

    G = numpy.empty(N//2 + 1, numpy.complex128)
    G[:N//2] = X * A + (numpy.roll(X[N//2-1::-1], 1)).conj() * B
    G[N//2] = X[0].real - X[0].imag

    return G


def prepare_rfft_input(arr):
    res = Type(dtypes.complex_for(arr.dtype), arr.shape[:-1] + (arr.shape[-1] // 2,))
    return Transformation(
        [
            Parameter('output', Annotation(res, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        ${input.ctype} re = ${input.load_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]} * 2);
        ${input.ctype} im = ${input.load_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]} * 2 + 1);
        ${output.store_same}(COMPLEX_CTR(${output.ctype})(re, im));
        """,
        connectors=['output'])


class RFFT(Computation):

    def __init__(self, arr_t, dont_store_last=False):
        self._dont_store_last = dont_store_last

        output_size = arr_t.shape[-1] // 2 + (0 if dont_store_last else 1)

        out_arr = Type(
            dtypes.complex_for(arr_t.dtype),
            arr_t.shape[:-1] + (output_size,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = input_.shape[-1]
        WNmk = numpy.exp(-2j * numpy.pi * numpy.arange(N//2) / N)
        A = 0.5 * (1 - 1j * WNmk)
        B = 0.5 * (1 + 1j * WNmk)

        A_arr = plan.persistent_array(A)
        B_arr = plan.persistent_array(B)

        cfft_arr = Type(output.dtype, input_.shape[:-1] + (input_.shape[-1] // 2,))
        cfft = FFT(cfft_arr, axes=(len(input_.shape) - 1,))

        prepare_input = prepare_rfft_input(input_)

        cfft.parameter.input.connect(
            prepare_input, prepare_input.output, real_input=prepare_input.input)

        temp = plan.temp_array_like(cfft.parameter.output)

        batch_size = helpers.product(output.shape[:-1])

        plan.computation_call(cfft, temp, input_)
        plan.kernel_call(
            TEMPLATE.get_def('prepare_rfft_output'),
                [output, temp, A_arr, B_arr],
                global_size=(batch_size, N // 2),
                render_kwds=dict(
                    slices=(len(input_.shape) - 1, 1),
                    N=N,
                    mul=functions.mul(output.dtype, output.dtype),
                    conj=functions.conj(output.dtype),
                    dont_store_last=self._dont_store_last))

        return plan


def get_multiply(output):
    return Transformation(
        [
            Parameter('output', Annotation(output, 'o')),
            Parameter('a', Annotation(output, 'i')),
            Parameter('b', Annotation(Type(output.dtype, (output.shape[-1],)), 'i'))
        ],
        """
        ${output.store_same}(${mul}(${a.load_same}, ${b.load_idx}(${idxs[-1]})));
        """,
        connectors=['output', 'a'],
        render_kwds=dict(mul=functions.mul(output.dtype, output.dtype))
        )


def get_prepare_rtfft_scan(output):
    return Transformation(
        [
            Parameter('output', Annotation(output, 'o')),
            Parameter('Y', Annotation(output, 'i')),
            Parameter('re_X_0', Annotation(
                Type(dtypes.real_for(output.dtype), output.shape[:-1]), 'i'))
        ],
        """
        ${Y.ctype} Y = ${Y.load_same};
        Y = COMPLEX_CTR(${Y.ctype})(Y.y, -Y.x);

        if (${idxs[-1]} == 0)
        {
            Y.x = Y.x / 2 + ${re_X_0.load_idx}(${", ".join(idxs[:-1])});
            Y.y /= 2;
        }

        ${output.store_same}(Y);
        """,
        connectors=['output', 'Y'],
        )


class RTFFT(Computation):
    """
    FFT of a real signal with translational anti-symmetry (x[k] = -x[N/2+k]).
    """

    def __init__(self, arr_t):

        out_arr = Type(
            dtypes.complex_for(arr_t.dtype),
            arr_t.shape[:-1] + (arr_t.shape[-1] // 2,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = input_.shape[-1] * 2
        batch_shape = input_.shape[:-1]
        batch_size = helpers.product(batch_shape)

        coeffs1 = 4 * numpy.sin(2 * numpy.pi * numpy.arange(N//2) / N)
        coeffs2 = 2 * numpy.cos(2 * numpy.pi * numpy.arange(N//2) / N)

        c1_arr = plan.persistent_array(coeffs1)
        c2_arr = plan.persistent_array(coeffs2)

        multiply = get_multiply(input_)

        # re_X_1 = sum(x * coeffs2)

        t = plan.temp_array_like(input_)
        rd = Reduce(t, predicate_sum(input_.dtype), axes=(len(input_.shape)-1,))

        rd.parameter.input.connect(
            multiply, multiply.output, x=multiply.a, c2=multiply.b)

        re_X_0 = plan.temp_array_like(rd.parameter.output)
        plan.computation_call(rd, re_X_0, input_, c2_arr)

        # Y = numpy.fft.rfft(x * coeffs1)

        rfft = RFFT(input_, dont_store_last=True)
        rfft.parameter.input.connect(
            multiply, multiply.output, x=multiply.a, c1=multiply.b)

        Y = plan.temp_array_like(rfft.parameter.output)
        plan.computation_call(rfft, Y, input_, c1_arr)

        # Y *= -1j
        # Y[0] /= 2
        # Y[0] += re_X_1
        # res = numpy.cumsum(Y[:-1])

        prepare_rtfft_scan = get_prepare_rtfft_scan(Y)

        sc = Scan(Y, predicate_sum(Y.dtype), axes=(-1,), exclusive=False)
        sc.parameter.input.connect(
            prepare_rtfft_scan, prepare_rtfft_scan.output,
            Y=prepare_rtfft_scan.Y, re_X_0=prepare_rtfft_scan.re_X_0)

        plan.computation_call(sc, output, Y, re_X_0)

        return plan
