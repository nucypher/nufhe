import numpy

from tfhe.keys import TFHEParameters
from tfhe.numeric_functions import Torus32

from tfhe.gpu_lwe import (
    LweKeySwitchTranslate_fromArray,
    LweKeySwitchKeyComputation,
    LweKeySwitchKeyComputation_ref,
    LweSymEncrypt, LweSymEncrypt_ref,
    LwePhase, LwePhase_ref
    )

from tfhe.numeric_functions import Torus32
import tfhe.random_numbers as rn


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
                        a[i,:] -= ks_a[l,j,x,:]
                        b[i] -= ks_b[l,j,x]
                        current_variances[i] += ks_current_variances[l,j,x]

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
    ks_a = numpy.random.randint(-1000, 1000, size=(outer_n, t, base, inner_n), dtype=Torus32)
    ks_b = numpy.random.randint(-1000, 1000, size=(outer_n, t, base), dtype=Torus32)
    ks_cv = numpy.random.normal(size=(outer_n, t, base))
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
    assert (b == b_test).all()
    assert numpy.allclose(cv, cv_test)


def test_LweKeySwitchKey(thread):

    numpy.random.seed(123)

    params = TFHEParameters()

    extracted_n = params.tgsw_params.tlwe_params.extracted_lweparams.n
    t = params.ks_t
    basebit = params.ks_basebit
    base = 1 << basebit
    inner_n = params.in_out_params.n
    alpha = params.tgsw_params.tlwe_params.alpha_min

    ks_a = numpy.empty((extracted_n, t, base, inner_n), dtype=Torus32)
    ks_b = numpy.empty((extracted_n, t, base), dtype=Torus32)
    ks_cv = numpy.empty((extracted_n, t, base), dtype=numpy.float64)

    in_key = numpy.random.randint(0, 2, size=extracted_n, dtype=numpy.int32)
    out_key = numpy.random.randint(0, 2, size=inner_n, dtype=numpy.int32)
    a_noises = numpy.random.randint(-2**31, 2**31, size=(extracted_n, t, base - 1, inner_n), dtype=Torus32)
    b_noises = numpy.random.normal(scale=params.in_out_params.alpha_min, size=(extracted_n, t, base - 1))

    test = LweKeySwitchKeyComputation(extracted_n, t, basebit, inner_n, alpha).compile(thread)
    ref = LweKeySwitchKeyComputation_ref(extracted_n, t, basebit, inner_n, alpha)

    ks_a_dev = thread.empty_like(ks_a)
    ks_b_dev = thread.empty_like(ks_b)
    ks_cv_dev = thread.empty_like(ks_cv)
    in_key_dev = thread.to_device(in_key)
    out_key_dev = thread.to_device(out_key)
    a_noises_dev = thread.to_device(a_noises)
    b_noises_dev = thread.to_device(b_noises)

    test(ks_a_dev, ks_b_dev, ks_cv_dev, in_key_dev, out_key_dev, a_noises_dev, b_noises_dev)
    ref(ks_a, ks_b, ks_cv, in_key, out_key, a_noises, b_noises)

    ks_a_test = ks_a_dev.get()
    ks_b_test = ks_b_dev.get()
    ks_cv_test = ks_cv_dev.get()

    assert (ks_a_test == ks_a).all()
    assert (ks_b_test == ks_b).all()
    assert numpy.allclose(ks_cv_test, ks_cv)


def test_LweSymEncrypt(thread):

    rng = numpy.random.RandomState(123)

    params = TFHEParameters()
    n = params.in_out_params.n
    alpha = params.tgsw_params.tlwe_params.alpha_min

    shape = (16, 20)
    result_a = numpy.empty(shape + (n,), numpy.int32)
    result_b = numpy.empty(shape, numpy.int32)
    result_cv = numpy.empty(shape, numpy.float64)
    key = rn._rand_uniform_int32(rng, (n,))
    messages = numpy.random.randint(-2**31, 2**31, size=shape, dtype=numpy.int32)
    noises_a = rn._rand_uniform_torus32(rng, messages.shape + (n,))
    noises_b = rn._rand_gaussian_torus32(rng, 0, alpha, messages.shape)

    test = LweSymEncrypt(shape, n, alpha).compile(thread)
    ref = LweSymEncrypt_ref(shape, n, alpha)

    result_a_dev = thread.empty_like(result_a)
    result_b_dev = thread.empty_like(result_b)
    result_cv_dev = thread.empty_like(result_cv)
    key_dev = thread.to_device(key)
    messages_dev = thread.to_device(messages)
    noises_a_dev = thread.to_device(noises_a)
    noises_b_dev = thread.to_device(noises_b)

    test(
        result_a_dev, result_b_dev, result_cv_dev,
        messages_dev, key_dev, noises_a_dev, noises_b_dev)
    ref(result_a, result_b, result_cv, messages, key, noises_a, noises_b)

    result_a_test = result_a_dev.get()
    result_b_test = result_b_dev.get()
    result_cv_test = result_cv_dev.get()

    assert (result_a_test == result_a).all()
    assert (result_b_test == result_b).all()
    assert numpy.allclose(result_cv_test, result_cv)


def test_LwePhase(thread):

    rng = numpy.random.RandomState(123)

    params = TFHEParameters()
    n = params.in_out_params.n

    shape = (16, 20)
    result = numpy.empty(shape, numpy.int32)
    a = rng.randint(-2**31, 2**31, size=shape + (n,), dtype=numpy.int32)
    b = rng.randint(-2**31, 2**31, size=shape, dtype=numpy.int32)
    key = rn._rand_uniform_int32(rng, (n,))

    test = LwePhase(shape, n).compile(thread)
    ref = LwePhase_ref(shape, n)

    result_dev = thread.empty_like(result)
    a_dev = thread.to_device(a)
    b_dev = thread.to_device(b)
    key_dev = thread.to_device(key)

    test(result_dev, a_dev, b_dev, key_dev)
    ref(result, a, b, key)

    result_test = result_dev.get()

    assert (result_test == result).all()
