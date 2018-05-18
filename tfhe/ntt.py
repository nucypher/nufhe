import numpy

from reikna.fft import FFT

import reikna.helpers as helpers
from reikna.cluda import dtypes
from reikna.core import Computation, Parameter, Annotation, Type, Transformation
import reikna.cluda.functions as functions

from tfhe import ntt_twiddle
#import ntt_twiddle

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

    def __init__(self, shape, i32_input=False, ntt_per_block=2):

        assert shape[-1] == 1024

        self._i32_input = i32_input
        self._ntt_per_block = ntt_per_block
        oarr = Type(numpy.uint64, shape)
        iarr = Type(numpy.int32, shape) if i32_input else oarr

        Computation.__init__(self, [
            Parameter('output', Annotation(oarr, 'o')),
            Parameter('input', Annotation(iarr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        twd = plan.persistent_array(ntt_twiddle.twd)
        twd_sqrt = plan.persistent_array(ntt_twiddle.twd_sqrt)

        batch_size = helpers.product(output.shape[:-1])
        blocks_num = helpers.min_blocks(batch_size, self._ntt_per_block)

        plan.kernel_call(
            TEMPLATE.get_def('ntt1024'),
                [output, input_, twd, twd_sqrt],
                global_size=(blocks_num, 128 * self._ntt_per_block),
                local_size=(1, 128 * self._ntt_per_block),
                render_kwds=dict(
                    i32_input=self._i32_input,
                    ntt_per_block=self._ntt_per_block,
                    batch_size=batch_size,
                    blocks_num=blocks_num,
                    slices=(len(output.shape) - 1, 1)))

        return plan


class NTTInv(Computation):

    def __init__(self, shape, i32_output=False, ntt_per_block=2):

        assert shape[-1] == 1024

        self._i32_output = i32_output
        self._ntt_per_block = ntt_per_block
        iarr = Type(numpy.uint64, shape)
        oarr = Type(numpy.int32, shape) if i32_output else iarr

        Computation.__init__(self, [
            Parameter('output', Annotation(oarr, 'o')),
            Parameter('input', Annotation(iarr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        twd_inv = plan.persistent_array(ntt_twiddle.twd_inv)
        twd_sqrt_inv = plan.persistent_array(ntt_twiddle.twd_sqrt_inv)

        batch_size = helpers.product(output.shape[:-1])
        blocks_num = helpers.min_blocks(batch_size, self._ntt_per_block)

        plan.kernel_call(
            TEMPLATE.get_def('ntt1024_inv'),
                [output, input_, twd_inv, twd_sqrt_inv],
                global_size=(blocks_num, 128 * self._ntt_per_block),
                local_size=(1, 128 * self._ntt_per_block),
                render_kwds=dict(
                    i32_output=self._i32_output,
                    ntt_per_block=self._ntt_per_block,
                    batch_size=batch_size,
                    blocks_num=blocks_num,
                    slices=(len(output.shape) - 1, 1)))

        return plan


class TLweFFTAddMulRTo_NTT(Computation):

    def __init__(self, tmpa_a, gsw):
        # tmpa_a: Complex, (batch, k+1, N)
        # decaFFT: Complex, (batch, k+1, l, N)
        # gsw: Complex, (n, k+1, l, k+1, N)

        N = tmpa_a.shape[-1]
        self._k = tmpa_a.shape[-2] - 1
        self._l = gsw.shape[-3]
        batch = tmpa_a.shape[:-2]

        assert len(batch) == 1

        decaFFT = Type(tmpa_a.dtype, batch + (self._k + 1, self._l, N))

        Computation.__init__(self,
            [Parameter('tmpa_a', Annotation(tmpa_a, 'o')),
            Parameter('decaFFT', Annotation(decaFFT, 'i')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('bk_idx', Annotation(numpy.int32))])


    def _build_plan(self, plan_factory, device_params, tmpa_a, decaFFT, gsw, bk_idx):

        plan = plan_factory()

        plan.kernel_call(
            TEMPLATE.get_def('TLweFFTAddMulRTo'),
            [tmpa_a, decaFFT, gsw, bk_idx],
            global_size=(helpers.product(tmpa_a.shape[:-2]),) + tuple(tmpa_a.shape[-2:]),
            render_kwds=dict(
                k=self._k, l=self._l,
                batch_len=len(tmpa_a.shape) - 2))

        return plan



class NTTMul(Computation):

    def __init__(self, r_shape, a_shape, b_shape):
        Computation.__init__(self,
            [Parameter('output', Annotation(Type(numpy.uint64, r_shape), 'o')),
            Parameter('a', Annotation(Type(numpy.uint64, a_shape), 'i')),
            Parameter('b', Annotation(Type(numpy.uint64, b_shape), 'i'))])

    def _build_plan(self, plan_factory, device_params, output, a, b):

        plan = plan_factory()

        plan.kernel_call(
            TEMPLATE.get_def('NTTMul'),
            [output, a, b],
            global_size=output.shape
            )

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

    batch = 8
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


def reference_poly_mul(p1, p2):
    N = p1.size

    result = numpy.empty(N, numpy.int32)

    for i in range(N):
        result[i] = (p1[:i+1] * p2[i::-1]).sum() - (p1[i+1:] * p2[:i:-1]).sum()

    return result


def test_ntt_poly_mul():

    from reikna.cluda import cuda_api

    api = cuda_api()
    thr = api.Thread.create()

    batch = 7
    a = numpy.random.randint(-2**31, 2**31, size=(batch, 1024), dtype=numpy.int32)
    b = numpy.random.randint(-2**10, 2**10, size=(batch, 1024), dtype=numpy.int32)

    c = NTT(a.shape, i32_input=True)
    ci = NTTInv(a.shape, i32_output=True)
    cc = c.compile(thr)
    cic = ci.compile(thr)

    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)

    af_dev = thr.empty_like(cc.parameter.output)
    bf_dev = thr.empty_like(cc.parameter.output)

    mul = NTTMul(a.shape, a.shape, a.shape)
    mulc = mul.compile(thr)

    cc(af_dev, a_dev)
    cc(bf_dev, b_dev)

    mulc(af_dev, af_dev, bf_dev)

    cic(a_dev, af_dev)

    ref = numpy.empty((batch, 1024), numpy.int32)
    for j in range(batch):
        ref[j] = reference_poly_mul(a[j], b[j])

    print((a_dev.get() == ref).all())


def reference(a):
    bnum = a.shape[0] // 4
    tnum = 512

    a = a.astype(numpy.uint64)

    tidx = numpy.tile(numpy.arange(tnum).reshape(1, tnum), (bnum, 1))
    t1d = tidx % 128
    r = numpy.empty((bnum, 512, 8), a.dtype)

    sh = a.reshape(bnum, 1024*4)
    for tid in range(512):
        ntt_id = tid // 128
        elem_id = tid % 128
        for i in range(8):
            r[:,tid,i] = sh[:,ntt_id*1024 + elem_id + i*128]

    for tid in range(512):
        ntt_id = tid // 128
        elem_id = tid % 128
        for i in range(8):
            sh[:,ntt_id*1024 + elem_id + i*128] = r[:,tid,i]

    return sh.reshape(a.shape)


def test_ntt_per_block():


    from reikna.cluda import cuda_api

    api = cuda_api()
    thr = api.Thread.create()

    batch = (500*2*2*16,)
    a = numpy.random.randint(0, 2**31, size=batch + (1024,), dtype=numpy.int32)
    a = numpy.tile(numpy.arange(1024).reshape(1, 1024), (batch[0], 1))

    c1 = NTT(a.shape, i32_input=True, ntt_per_block=1).compile(thr)
    ci1 = NTTInv(a.shape, i32_output=True, ntt_per_block=1).compile(thr)
    c4 = NTT(a.shape, i32_input=True, ntt_per_block=4).compile(thr)
    ci4 = NTTInv(a.shape, i32_output=True, ntt_per_block=4).compile(thr)

    a_dev = thr.to_device(a)
    af_dev = thr.empty_like(c1.parameter.output)

    c1(af_dev, a_dev)
    af1 = af_dev.get()

    c4(af_dev, a_dev)
    af4 = af_dev.get()

    assert (af1 == af4).all()

    ci1(a_dev, af_dev)
    a1 = a_dev.get()
    ci4(a_dev, af_dev)
    a4 = a_dev.get()

    print("inversion is correct:", (a1 == a).all())

    for i in range(batch[0]):
        if not (a1[i] == a4[i]).all():
            print(i, i % 4, (a1[i] != a4[i]).sum())
            print(numpy.where(a1[i] != a4[i]))
            break

    assert (a1 == a4).all()


if __name__ == '__main__':
    test_ntt()
    test_ntt_i32()
    #generate_twiddle_factors()
    test_ntt_poly_mul()
    test_ntt_per_block()
