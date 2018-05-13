import numpy

from reikna.fft import FFT

import reikna.helpers as helpers
from reikna.cluda import dtypes
from reikna.core import Computation, Parameter, Annotation, Type, Transformation
import reikna.cluda.functions as functions

from tfhe import ntt_twiddle


TEMPLATE = helpers.template_for(__file__)


class NTTTwiddleFactors(Computation):

    def __init__(self):
        res = Type(numpy.uint64, 1024)
        Computation.__init__(self, [
            Parameter('twd', Annotation(res, 'o')),
            Parameter('twd_inv', Annotation(res, 'o')),
            Parameter('twd_sqrt', Annotation(res, 'o')),
            Parameter('twd_sqrt_inv', Annotation(res, 'o')),
            ])

    def _build_plan(self, plan_factory, device_params, twd, twd_inv, twd_sqrt, twd_sqrt_inv):
        plan = plan_factory()

        plan.kernel_call(
            TEMPLATE.get_def('ntt1024_twiddle'),
                [twd, twd_inv],
                global_size=(2, 8, 8),
                local_size=(2, 8, 8))

        plan.kernel_call(
            TEMPLATE.get_def('ntt1024_twiddle_sqrt'),
                [twd_sqrt, twd_sqrt_inv],
                global_size=(16 * 64,))

        return plan


class NTT(Computation):

    def __init__(self, shape, i32_input=False):

        assert shape[-1] == 1024

        self._i32_input = i32_input
        oarr = Type(numpy.uint64, shape)
        iarr = Type(numpy.int32, shape) if i32_input else oarr

        Computation.__init__(self, [
            Parameter('output', Annotation(oarr, 'o')),
            Parameter('input', Annotation(iarr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        twd = plan.persistent_array(ntt_twiddle.twd)
        twd_sqrt = plan.persistent_array(ntt_twiddle.twd_sqrt)

        plan.kernel_call(
            TEMPLATE.get_def('ntt1024'),
                [output, input_, twd, twd_sqrt],
                global_size=(helpers.product(output.shape[:-1]), 128),
                local_size=(1, 128),
                render_kwds=dict(
                    i32_input=self._i32_input,
                    slices=(len(output.shape) - 1, 1)))

        return plan


class NTTInv(Computation):

    def __init__(self, shape, i32_output=False):

        assert shape[-1] == 1024

        self._i32_output = i32_output
        iarr = Type(numpy.uint64, shape)
        oarr = Type(numpy.int32, shape) if i32_output else iarr

        Computation.__init__(self, [
            Parameter('output', Annotation(oarr, 'o')),
            Parameter('input', Annotation(iarr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        twd_inv = plan.persistent_array(ntt_twiddle.twd_inv)
        twd_sqrt_inv = plan.persistent_array(ntt_twiddle.twd_sqrt_inv)

        plan.kernel_call(
            TEMPLATE.get_def('ntt1024_inv'),
                [output, input_, twd_inv, twd_sqrt_inv],
                global_size=(helpers.product(output.shape[:-1]), 128),
                local_size=(1, 128),
                render_kwds=dict(
                    i32_output=self._i32_output,
                    slices=(len(output.shape) - 1, 1)))

        return plan


def generate_twiddle_factors():

    from reikna.cluda import cuda_api

    api = cuda_api()
    thr = api.Thread.create()
    twiddles = NTTTwiddleFactors().compile(thr)

    twd = thr.empty_like(twiddles.parameter.twd)
    twd_inv = thr.empty_like(twiddles.parameter.twd_inv)
    twd_sqrt = thr.empty_like(twiddles.parameter.twd_sqrt)
    twd_sqrt_inv = thr.empty_like(twiddles.parameter.twd_sqrt_inv)

    twiddles(twd, twd_inv, twd_sqrt, twd_sqrt_inv)


def test_ntt():

    from reikna.cluda import cuda_api

    api = cuda_api()
    thr = api.Thread.create()

    batch = (4, 6)
    a = numpy.random.randint(0, 2**64 - 2**32 + 1, size=batch + (1024,), dtype=numpy.uint64)

    c = NTT(a.shape)
    ci = NTTInv(a.shape)
    cc = c.compile(thr)
    cic = ci.compile(thr)

    a_dev = thr.to_device(a)
    cc(a_dev, a_dev)
    cic(a_dev, a_dev)

    print((a_dev.get() == a).all())


def test_ntt_i32():

    from reikna.cluda import cuda_api

    api = cuda_api()
    thr = api.Thread.create()

    batch = 10
    a = numpy.random.randint(-2**31, 2**31, size=(batch, 1024), dtype=numpy.int32)

    c = NTT(a.shape, i32_input=True)
    ci = NTTInv(a.shape, i32_output=True)
    cc = c.compile(thr)
    cic = ci.compile(thr)

    a_dev = thr.to_device(a)
    a_u64 = thr.empty_like(cc.parameter.output)
    cc(a_u64, a_dev)
    cic(a_dev, a_u64)

    print((a_dev.get() == a).all())


if __name__ == '__main__':
    test_ntt()
    test_ntt_i32()
    generate_twiddle_factors()
