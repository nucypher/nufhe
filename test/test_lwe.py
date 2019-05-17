# Copyright (C) 2018 NuCypher
#
# This file is part of nufhe.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pytest
import numpy

from nufhe.api_low_level import NuFHEParameters
from nufhe.lwe import LweSampleArrayShapeInfo, LweSampleArray, concatenate
from nufhe.lwe_gpu import (
    LweKeyswitch,
    MakeLweKeyswitchKey,
    LweEncrypt,
    LweDecrypt,
    LweLinear,
    LweNoiselessTrivial,
    LweNoiselessTrivialConstant,
    )
from nufhe.lwe_cpu import (
    LweKeyswitchReference,
    MakeLweKeyswitchKeyReference,
    LweEncryptReference,
    LweDecryptReference,
    LweLinearReference,
    LweNoiselessTrivialReference,
    LweNoiselessTrivialConstantReference,
    )
from nufhe.numeric_functions import Torus32, Int32, ErrorFloat, double_to_t32
import nufhe.random_numbers as rn

from utils import get_test_array, errors_allclose


def test_lwe_keyswitch(thread):

    batch_shape = (4, 5)

    params = NuFHEParameters()
    input_size = params.tgsw_params.tlwe_params.extracted_lweparams.size
    output_size = params.in_out_params.size
    decomp_length = params.ks_decomp_length
    log2_base = params.ks_log2_base
    base = 2**log2_base

    result_a = numpy.empty(batch_shape + (output_size,), Torus32)
    result_b = numpy.empty(batch_shape, Torus32)
    result_cv = numpy.empty(batch_shape, ErrorFloat)
    ks_a = get_test_array((input_size, decomp_length, base, output_size), Torus32, (-1000, 1000))
    ks_b = get_test_array((input_size, decomp_length, base), Torus32, (-1000, 1000))
    ks_cv = get_test_array((input_size, decomp_length, base), ErrorFloat, (-1, 1))

    # The base=0 slice of the keyswitch key is a "padding" - it's filled with zeroes.
    # The keyswitch function may rely on that.
    ks_a[:,:,0,:] = 0
    ks_b[:,:,0] = 0
    ks_cv[:,:,0] = 0

    source_a = get_test_array(batch_shape + (input_size,), Torus32)
    source_b = get_test_array(batch_shape, Torus32, (-1000, 1000))

    result_a_dev = thread.empty_like(result_a)
    result_b_dev = thread.empty_like(result_b)
    result_cv_dev = thread.empty_like(result_cv)
    ks_a_dev = thread.to_device(ks_a)
    ks_b_dev = thread.to_device(ks_b)
    ks_cv_dev = thread.to_device(ks_cv)
    source_a_dev = thread.to_device(source_a)
    source_b_dev = thread.to_device(source_b)

    shape_info = LweSampleArrayShapeInfo(result_a_dev, result_b_dev, result_cv_dev)
    test = LweKeyswitch(
        shape_info, input_size, output_size, decomp_length, log2_base).compile(thread)
    ref = LweKeyswitchReference(
        shape_info, input_size, output_size, decomp_length, log2_base)

    test(
        result_a_dev, result_b_dev, result_cv_dev,
        ks_a_dev, ks_b_dev, ks_cv_dev,
        source_a_dev, source_b_dev)
    result_a_test = result_a_dev.get()
    result_b_test = result_b_dev.get()
    result_cv_test = result_cv_dev.get()

    ref(result_a, result_b, result_cv, ks_a, ks_b, ks_cv, source_a, source_b)

    assert (result_a == result_a_test).all()
    assert (result_b == result_b_test).all()
    assert errors_allclose(result_cv, result_cv_test)


def test_make_lwe_keyswitch_key(thread):

    params = NuFHEParameters()
    input_size = params.tgsw_params.tlwe_params.extracted_lweparams.size
    output_size = params.in_out_params.size
    decomp_length = params.ks_decomp_length
    log2_base = params.ks_log2_base
    base = 2**log2_base
    noise = params.in_out_params.min_noise

    ks_a = numpy.empty((input_size, decomp_length, base, output_size), dtype=Torus32)
    ks_b = numpy.empty((input_size, decomp_length, base), dtype=Torus32)
    ks_cv = numpy.empty((input_size, decomp_length, base), dtype=ErrorFloat)

    in_key = get_test_array(input_size, Int32, (0, 2))
    out_key = get_test_array(output_size, Int32, (0, 2))
    noises_a = get_test_array((input_size, decomp_length, base - 1, output_size), Torus32)
    noises_b = double_to_t32(
        get_test_array((input_size, decomp_length, base - 1), ErrorFloat, (-noise, noise)))

    test = MakeLweKeyswitchKey(
        input_size, output_size, decomp_length, log2_base, noise).compile(thread)
    ref = MakeLweKeyswitchKeyReference(
        input_size, output_size, decomp_length, log2_base, noise)

    ks_a_dev = thread.empty_like(ks_a)
    ks_b_dev = thread.empty_like(ks_b)
    ks_cv_dev = thread.empty_like(ks_cv)
    in_key_dev = thread.to_device(in_key)
    out_key_dev = thread.to_device(out_key)
    noises_a_dev = thread.to_device(noises_a)
    noises_b_dev = thread.to_device(noises_b)

    test(ks_a_dev, ks_b_dev, ks_cv_dev, in_key_dev, out_key_dev, noises_a_dev, noises_b_dev)
    ref(ks_a, ks_b, ks_cv, in_key, out_key, noises_a, noises_b)

    ks_a_test = ks_a_dev.get()
    ks_b_test = ks_b_dev.get()
    ks_cv_test = ks_cv_dev.get()

    assert (ks_a_test == ks_a).all()
    assert (ks_b_test == ks_b).all()
    assert errors_allclose(ks_cv_test, ks_cv)


def test_lwe_encrypt(thread):

    params = NuFHEParameters()
    lwe_size = params.in_out_params.size
    noise = params.in_out_params.min_noise

    shape = (16, 20)
    result_a = numpy.empty(shape + (lwe_size,), Torus32)
    result_b = numpy.empty(shape, Torus32)
    result_cv = numpy.empty(shape, ErrorFloat)
    key = get_test_array(lwe_size, Int32, (0, 2))
    messages = get_test_array(shape, Torus32)
    noises_a = get_test_array(shape + (lwe_size,), Torus32)
    noises_b = get_test_array(shape, Torus32)

    test = LweEncrypt(shape, lwe_size, noise).compile(thread)
    ref = LweEncryptReference(shape, lwe_size, noise)

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
    assert errors_allclose(result_cv_test, result_cv)


def test_lwe_decrypt(thread):

    params = NuFHEParameters()
    lwe_size = params.in_out_params.size

    shape = (16, 20)
    result = numpy.empty(shape, Torus32)
    lwe_a = get_test_array(shape + (lwe_size,), Torus32)
    lwe_b = get_test_array(shape, Torus32)
    key = get_test_array(lwe_size, Int32, (0, 2))

    test = LweDecrypt(shape, lwe_size).compile(thread)
    ref = LweDecryptReference(shape, lwe_size)

    result_dev = thread.empty_like(result)
    lwe_a_dev = thread.to_device(lwe_a)
    lwe_b_dev = thread.to_device(lwe_b)
    key_dev = thread.to_device(key)

    test(result_dev, lwe_a_dev, lwe_b_dev, key_dev)
    ref(result, lwe_a, lwe_b, key)

    result_test = result_dev.get()

    assert (result_test == result).all()


@pytest.mark.parametrize('positive_coeff', [False, True], ids=['p<0', 'p>0'])
@pytest.mark.parametrize('add_result', [False, True], ids=['replace_result', 'update_result'])
def test_lwe_linear(thread, positive_coeff, add_result):

    params = NuFHEParameters()
    lwe_size = params.in_out_params.size

    shape = (10, 20)

    res_a = get_test_array(shape + (lwe_size,), Torus32)
    res_b = get_test_array(shape, Torus32)
    res_cv = get_test_array(shape, ErrorFloat, (-1, 1))

    src_a = get_test_array(shape + (lwe_size,), Torus32)
    src_b = get_test_array(shape, Torus32)
    src_cv = get_test_array(shape, ErrorFloat, (-1, 1))

    coeff = 1 if positive_coeff else -1

    shape_info = LweSampleArrayShapeInfo(src_a, src_b, src_cv)

    test = LweLinear(shape_info, shape_info, add_result=add_result).compile(thread)
    ref = LweLinearReference(shape_info, shape_info, add_result=add_result)

    res_a_dev = thread.to_device(res_a)
    res_b_dev = thread.to_device(res_b)
    res_cv_dev = thread.to_device(res_cv)
    src_a_dev = thread.to_device(src_a)
    src_b_dev = thread.to_device(src_b)
    src_cv_dev = thread.to_device(src_cv)
    thread.synchronize()

    test(res_a_dev, res_b_dev, res_cv_dev, src_a_dev, src_b_dev, src_cv_dev, coeff)
    ref(res_a, res_b, res_cv, src_a, src_b, src_cv, coeff)

    assert (res_a_dev.get() == res_a).all()
    assert (res_b_dev.get() == res_b).all()
    assert errors_allclose(res_cv_dev.get(), res_cv)


def test_lwe_linear_broadcast(thread):

    params = NuFHEParameters()
    lwe_size = params.in_out_params.size

    res_shape = (10, 20)
    src_shape = res_shape[1:]

    res_a = get_test_array(res_shape + (lwe_size,), Torus32)
    res_b = get_test_array(res_shape, Torus32)
    res_cv = get_test_array(res_shape, ErrorFloat, (-1, 1))

    src_a = get_test_array(src_shape + (lwe_size,), Torus32)
    src_b = get_test_array(src_shape, Torus32)
    src_cv = get_test_array(src_shape, ErrorFloat, (-1, 1))

    coeff = 1
    add_result = True

    res_shape_info = LweSampleArrayShapeInfo(res_a, res_b, res_cv)
    src_shape_info = LweSampleArrayShapeInfo(src_a, src_b, src_cv)

    test = LweLinear(res_shape_info, src_shape_info, add_result=add_result).compile(thread)
    ref = LweLinearReference(res_shape_info, src_shape_info, add_result=add_result)

    res_a_dev = thread.to_device(res_a)
    res_b_dev = thread.to_device(res_b)
    res_cv_dev = thread.to_device(res_cv)
    src_a_dev = thread.to_device(src_a)
    src_b_dev = thread.to_device(src_b)
    src_cv_dev = thread.to_device(src_cv)
    thread.synchronize()

    test(res_a_dev, res_b_dev, res_cv_dev, src_a_dev, src_b_dev, src_cv_dev, coeff)
    ref(res_a, res_b, res_cv, src_a, src_b, src_cv, coeff)

    assert (res_a_dev.get() == res_a).all()
    assert (res_b_dev.get() == res_b).all()
    assert errors_allclose(res_cv_dev.get(), res_cv)


def test_lwe_noiseless_trivial_constant(thread):

    params = NuFHEParameters()
    lwe_size = params.in_out_params.size

    shape = (10, 20)

    res_a = numpy.empty(shape + (lwe_size,), Torus32)
    res_b = numpy.empty(shape, Torus32)
    res_cv = numpy.empty(shape, ErrorFloat)
    mu = Torus32(-5)

    shape_info = LweSampleArrayShapeInfo(res_a, res_b, res_cv)

    test = LweNoiselessTrivialConstant(shape_info).compile(thread)
    ref = LweNoiselessTrivialConstantReference(shape_info)

    res_a_dev = thread.empty_like(res_a)
    res_b_dev = thread.empty_like(res_b)
    res_cv_dev = thread.empty_like(res_cv)

    test(res_a_dev, res_b_dev, res_cv_dev, mu)
    ref(res_a, res_b, res_cv, mu)

    assert (res_a_dev.get() == res_a).all()
    assert (res_b_dev.get() == res_b).all()
    assert errors_allclose(res_cv_dev.get(), res_cv)


def test_lwe_noiseless_trivial(thread):

    params = NuFHEParameters()
    lwe_size = params.in_out_params.size

    shape = (10, 20)

    res_a = numpy.empty(shape + (lwe_size,), Torus32)
    res_b = numpy.empty(shape, Torus32)
    res_cv = numpy.empty(shape, ErrorFloat)
    mus = get_test_array(shape, Torus32)

    shape_info = LweSampleArrayShapeInfo(res_a, res_b, res_cv)

    test = LweNoiselessTrivial(shape_info, shape).compile(thread)
    ref = LweNoiselessTrivialReference(shape_info, shape)

    res_a_dev = thread.empty_like(res_a)
    res_b_dev = thread.empty_like(res_b)
    res_cv_dev = thread.empty_like(res_cv)
    mus_dev = thread.to_device(mus)

    test(res_a_dev, res_b_dev, res_cv_dev, mus_dev)
    ref(res_a, res_b, res_cv, mus)

    assert (res_a_dev.get() == res_a).all()
    assert (res_b_dev.get() == res_b).all()
    assert errors_allclose(res_cv_dev.get(), res_cv)


@pytest.mark.parametrize('src_len', [0, 1])
def test_lwe_noiseless_trivial_broadcast(thread, src_len):

    params = NuFHEParameters()
    lwe_size = params.in_out_params.size

    res_shape = (10, 20)
    src_shape = res_shape[len(res_shape)-src_len:]

    res_a = numpy.empty(res_shape + (lwe_size,), Torus32)
    res_b = numpy.empty(res_shape, Torus32)
    res_cv = numpy.empty(res_shape, ErrorFloat)
    mus = get_test_array(src_shape, Torus32)

    shape_info = LweSampleArrayShapeInfo(res_a, res_b, res_cv)

    test = LweNoiselessTrivial(shape_info, src_shape).compile(thread)
    ref = LweNoiselessTrivialReference(shape_info, src_shape)

    res_a_dev = thread.empty_like(res_a)
    res_b_dev = thread.empty_like(res_b)
    res_cv_dev = thread.empty_like(res_cv)
    mus_dev = thread.to_device(mus)

    test(res_a_dev, res_b_dev, res_cv_dev, mus_dev)
    ref(res_a, res_b, res_cv, mus)

    assert (res_a_dev.get() == res_a).all()
    assert (res_b_dev.get() == res_b).all()
    assert errors_allclose(res_cv_dev.get(), res_cv)


def mock_ciphertext(thread, params, shape):
    ciphertext = LweSampleArray.empty(thread, params, shape)
    ciphertext.a = thread.to_device(get_test_array(ciphertext.a.shape, ciphertext.a.dtype))
    ciphertext.b = thread.to_device(get_test_array(ciphertext.b.shape, ciphertext.b.dtype))
    ciphertext.current_variances = thread.to_device(
        get_test_array(ciphertext.current_variances.shape, ciphertext.current_variances.dtype))
    return ciphertext


def test_lwe_copy(thread):

    params = NuFHEParameters()
    lwe_params = params.in_out_params

    shape = (3, 4, 5)

    ciphertext = mock_ciphertext(thread, lwe_params, shape)
    ciphertext_copy = ciphertext.copy()

    assert ciphertext == ciphertext_copy
    assert ciphertext.a is not ciphertext_copy.a
    assert ciphertext.b is not ciphertext_copy.b
    assert ciphertext.current_variances is not ciphertext_copy.current_variances


@pytest.mark.parametrize('shift', [7, -9, 0])
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_lwe_roll(thread, shift, axis):

    params = NuFHEParameters()
    lwe_params = params.in_out_params

    shape = (3, 4, 5)

    ciphertext = mock_ciphertext(thread, lwe_params, shape)

    ciphertext_rolled = ciphertext.copy()
    ciphertext_rolled.roll(shift, axis=axis)

    src_a = ciphertext.a.get()
    src_b = ciphertext.b.get()
    src_cv = ciphertext.current_variances.get()

    res_a = ciphertext_rolled.a.get()
    res_b = ciphertext_rolled.b.get()
    res_cv = ciphertext_rolled.current_variances.get()

    roll_axis = axis % len(shape)

    assert (numpy.roll(src_a, shift, roll_axis) == res_a).all()
    assert (numpy.roll(src_b, shift, roll_axis) == res_b).all()
    assert numpy.allclose(numpy.roll(src_cv, shift, roll_axis), res_cv)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('out_none', [False, True])
def test_lwe_concatenate(thread, axis, out_none):

    params = NuFHEParameters()
    lwe_params = params.in_out_params

    if axis == 0:
        shapes = [(3, 4), (1, 4), (4, 4)]
    elif axis == 1:
        shapes = [(4, 3), (4, 1), (4, 4)]

    ciphertexts = [mock_ciphertext(thread, lwe_params, shape) for shape in shapes]

    if out_none:
        out = None
    else:
        out = mock_ciphertext(thread, lwe_params, (8, 4) if axis == 0 else (4, 8))

    out = concatenate(ciphertexts, axis=axis, out=out)

    ref_a = numpy.concatenate([ciphertext.a.get() for ciphertext in ciphertexts], axis=axis)
    ref_b = numpy.concatenate([ciphertext.b.get() for ciphertext in ciphertexts], axis=axis)
    ref_cv = numpy.concatenate(
        [ciphertext.current_variances.get() for ciphertext in ciphertexts], axis=axis)

    assert (out.a.get() == ref_a).all()
    assert (out.b.get() == ref_b).all()
    assert numpy.allclose(out.current_variances.get(), ref_cv)


class _GetSlices:
    def __getitem__(self, index):
        return index

_get_slices = _GetSlices()

_lwe_assign_tests = [
    ((3, 4), _get_slices[1:], (3, 4), _get_slices[:-1], "[contig]=[contig]"),
    ((10,), _get_slices[1:10:2], (10,), _get_slices[:10:2], "[discontig]=[discontig]"),
    ((5,), _get_slices[1], (5,), _get_slices[2], "[scalar]=[scalar]"),
]

@pytest.mark.parametrize(
    'lwe_assign_test',
    [test[:-1] for test in _lwe_assign_tests],
    ids=[test[-1] for test in _lwe_assign_tests])
def test_lwe_assign(thread, lwe_assign_test):

    src_shape, src_slice, dst_shape, dst_slice = lwe_assign_test

    params = NuFHEParameters()
    lwe_params = params.in_out_params

    ct_src = mock_ciphertext(thread, lwe_params, src_shape)
    ct_dst = mock_ciphertext(thread, lwe_params, dst_shape)

    ref_a = ct_dst.a.get()
    ref_b = ct_dst.b.get()
    ref_cv = ct_dst.current_variances.get()

    ct_dst[dst_slice] = ct_src[src_slice]

    ref_a[dst_slice] = ct_src.a.get()[src_slice]
    ref_b[dst_slice] = ct_src.b.get()[src_slice]
    ref_cv[dst_slice] = ct_src.current_variances.get()[src_slice]

    assert (ct_dst.a.get() == ref_a).all()
    assert (ct_dst.b.get() == ref_b).all()
    assert numpy.allclose(ct_dst.current_variances.get(), ref_cv)
