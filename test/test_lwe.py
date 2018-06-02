import numpy

from tfhe.keys import TFHEParameters
from tfhe.numeric_functions import Torus32

from tfhe.gpu_lwe import LweKeySwitchTranslate_fromArray

from tfhe.numeric_functions import Torus32


def LweKeySwitchTranslate_fromArray_reference(
        batch_shape, t: int, outer_n, inner_n, basebit: int):

    def _kernel(a, b, current_variances, ks_a, ks_b, ks_current_variances, ai):

        base = 1 << basebit # base=2 in [CGGI16]
        prec_offset = 1 << (32 - (1 + basebit * t)) # precision
        mask = base - 1

        js = numpy.arange(1, t+1).reshape(1, 1, t)
        ai = ai.reshape(ai.shape + (1,))
        aijs = (((ai + prec_offset) >> (32 - js * basebit)) & mask)

        for i in range(batch_shape[0]):
            for l in range(outer_n):
                for j in range(t):
                    x = aijs[i,l,j]
                    if x != 0:
                        a[i,:] -= ks_a[:,x,l,j]
                        b[i] -= ks_b[x,l,j]
                        current_variances[i] += ks_current_variances[x,l,j]

    return _kernel


def test_LweKeySwitchTranslate_fromArray(thread):

    numpy.random.seed(123)

    batch_shape = (1,)

    params = TFHEParameters()
    tgsw_params = params.tgsw_params
    outer_n = tgsw_params.tlwe_params.extracted_lweparams.n
    inner_n = params.in_out_params.n
    t = params.ks_t
    basebit = params.ks_basebit
    base = 1 << basebit

    a = numpy.random.randint(-1000, 1000, size=batch_shape + (inner_n,), dtype=Torus32)
    b = numpy.random.randint(-1000, 1000, size=batch_shape, dtype=Torus32)
    cv = numpy.random.normal(size=batch_shape)
    ks_a = numpy.random.randint(-1000, 1000, size=(inner_n, base, outer_n, t), dtype=Torus32)
    ks_b = numpy.random.randint(-1000, 1000, size=(base, outer_n, t), dtype=Torus32)
    ks_cv = numpy.random.normal(size=(base, outer_n, t))
    ai = numpy.random.randint(-2**31, 2**31, batch_shape + (outer_n,), dtype=Torus32)

    a_dev = thread.to_device(a)
    b_dev = thread.to_device(b)
    cv_dev = thread.to_device(cv)
    ks_a_dev = thread.to_device(ks_a)
    ks_b_dev = thread.to_device(ks_b)
    ks_cv_dev = thread.to_device(ks_cv)
    ai_dev = thread.to_device(ai)

    test = LweKeySwitchTranslate_fromArray(batch_shape, t, outer_n, inner_n, basebit).compile(thread)
    ref = LweKeySwitchTranslate_fromArray_reference(batch_shape, t, outer_n, inner_n, basebit)

    test(a_dev, b_dev, cv_dev, ks_a_dev, ks_b_dev, ks_cv_dev, ai_dev)
    a_test = a_dev.get()
    b_test = b_dev.get()
    cv_test = cv_dev.get()

    ref(a, b, cv, ks_a, ks_b, ks_cv, ai)

    assert (a == a_test).all()
    #assert (b == b_test).all()
    #assert numpy.allclose(cv, cv_test)
