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


def prepare_irfft_output(arr):
    res = Type(dtypes.real_for(arr.dtype), arr.shape[:-1] + (arr.shape[-1] * 2,))
    return Transformation(
        [
            Parameter('output', Annotation(res, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        ${input.ctype} x = ${input.load_same};
        ${output.store_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]} * 2, x.x);
        ${output.store_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]} * 2 + 1, x.y);
        """,
        connectors=['output'])


class IRFFT(Computation):

    def __init__(self, arr_t):

        output_size = (arr_t.shape[-1] - 1) * 2

        out_arr = Type(
            dtypes.real_for(arr_t.dtype),
            arr_t.shape[:-1] + (output_size,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = (input_.shape[-1] - 1) * 2

        WNmk = numpy.exp(-2j * numpy.pi * numpy.arange(N//2) / N)
        A = 0.5 * (1 - 1j * WNmk)
        B = 0.5 * (1 + 1j * WNmk)

        A_arr = plan.persistent_array(A.conj())
        B_arr = plan.persistent_array(B.conj())

        cfft_arr = Type(input_.dtype, input_.shape[:-1] + (N // 2,))
        cfft = FFT(cfft_arr, axes=(len(input_.shape) - 1,))

        prepare_output = prepare_irfft_output(cfft.parameter.output)

        cfft.parameter.output.connect(
            prepare_output, prepare_output.input, real_output=prepare_output.output)

        temp = plan.temp_array_like(cfft.parameter.input)

        batch_size = helpers.product(output.shape[:-1])

        plan.kernel_call(
            TEMPLATE.get_def('prepare_irfft_input'),
                [temp, input_, A_arr, B_arr],
                global_size=(batch_size, N // 2),
                render_kwds=dict(
                    slices=(len(input_.shape) - 1, 1),
                    N=N,
                    mul=functions.mul(input_.dtype, input_.dtype),
                    conj=functions.conj(input_.dtype)))

        plan.computation_call(cfft, output, temp, inverse=True)

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


def get_prepare_irtfft_input(X):
    # Input: size N//4
    # Output: size N//4+1

    N = X.shape[-1] * 4
    Y = Type(X.dtype, X.shape[:-1] + (N // 4 + 1,))

    return Transformation(
        [
            Parameter('Y', Annotation(Y, 'o')),
            Parameter('X', Annotation(X, 'i')),
        ],
        """
        ${Y.ctype} Y;
        if (${idxs[-1]} == 0)
        {
            ${X.ctype} X = ${X.load_idx}(${", ".join(idxs[:-1])}, 0);
            Y = COMPLEX_CTR(${Y.ctype})(-2 * X.y, 0);
        }
        else if (${idxs[-1]} == ${N//4})
        {
            ${X.ctype} X = ${X.load_idx}(${", ".join(idxs[:-1])}, ${N//4-1});
            Y = COMPLEX_CTR(${Y.ctype})(2 * X.y, 0);
        }
        else
        {
            ${X.ctype} X = ${X.load_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]});
            ${X.ctype} X_prev = ${X.load_idx}(${", ".join(idxs[:-1])}, ${idxs[-1]} - 1);
            ${X.ctype} diff = X - X_prev;
            Y = COMPLEX_CTR(${Y.ctype})(-diff.y, diff.x);
        }

        ${Y.store_same}(Y);
        """,
        connectors=['Y'],
        render_kwds=dict(N=N)
        )


def get_prepare_irtfft_output(y):
    # Input: size N//4
    # Output: size N//4

    N = y.shape[-1] * 2

    return Transformation(
        [
            Parameter('x', Annotation(y, 'o')),
            Parameter('y', Annotation(y, 'i')),
            Parameter('x0', Annotation(Type(y.dtype, y.shape[:-1]), 'i')),
            Parameter('coeffs', Annotation(Type(y.dtype, (N//2,)), 'i')),
        ],
        """
        ${y.ctype} y = ${y.load_same};
        ${coeffs.ctype} coeff = ${coeffs.load_idx}(${idxs[-1]});

        ${x.ctype} x;

        if (${idxs[-1]} == 0)
        {
            ${x0.ctype} x0 = ${x0.load_idx}(${", ".join(idxs[:-1])});
            x = x0 / ${N // 2};
        }
        else
        {
            x = y * coeff;
        }

        ${x.store_same}(x);
        """,
        connectors=['y'],
        render_kwds=dict(N=N)
        )


class IRTFFT(Computation):
    """
    IFFT of a real signal with translational anti-symmetry (x[k] = -x[N/2+k]).
    """

    def __init__(self, arr_t):

        out_arr = Type(
            dtypes.real_for(arr_t.dtype),
            arr_t.shape[:-1] + (arr_t.shape[-1] * 2,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = input_.shape[-1] * 4
        batch_shape = input_.shape[:-1]
        batch_size = helpers.product(batch_shape)

        # The first element is unused
        coeffs = numpy.concatenate(
            [[0], 1 / (4 * numpy.sin(2 * numpy.pi * numpy.arange(1, N//2) / N))])
        coeffs_arr = plan.persistent_array(coeffs)

        prepare_irtfft_input = get_prepare_irtfft_input(input_)
        prepare_irtfft_output = get_prepare_irtfft_output(output)

        irfft = IRFFT(prepare_irtfft_input.Y)
        irfft.parameter.input.connect(
            prepare_irtfft_input, prepare_irtfft_input.Y,
            X=prepare_irtfft_input.X)
        irfft.parameter.output.connect(
            prepare_irtfft_output, prepare_irtfft_output.y,
            x=prepare_irtfft_output.x,
            x0=prepare_irtfft_output.x0, coeffs=prepare_irtfft_output.coeffs)

        real = Transformation(
            [
                Parameter('output', Annotation(Type(dtypes.real_for(input_.dtype), input_.shape), 'o')),
                Parameter('input', Annotation(input_, 'i')),
            ],
            """
            ${output.store_same}((${input.load_same}).x);
            """,
            connectors=['output']
            )

        rd_t = Type(output.dtype, input_.shape)
        rd = Reduce(rd_t, predicate_sum(rd_t.dtype), axes=(len(input_.shape)-1,))
        rd.parameter.input.connect(real, real.output, X=real.input)

        x0 = plan.temp_array_like(rd.parameter.output)

        plan.computation_call(rd, x0, input_)
        plan.computation_call(irfft, output, x0, coeffs_arr, input_)

        return plan
