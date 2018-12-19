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

import numpy

from nufhe.api_low_level import NuFHEParameters
from nufhe.numeric_functions import Torus32, Int32, ErrorFloat
from nufhe.tlwe import TLweParams
from nufhe.tlwe_gpu import (
    TLweNoiselessTrivial,
    TLweExtractLweSamples,
    TLweEncryptZero,
    )
from nufhe.tlwe_cpu import (
    TLweNoiselessTrivialReference,
    TLweExtractLweSamplesReference,
    TLweEncryptZeroReference,
    )
from nufhe import PerformanceParameters

from utils import get_test_array, errors_allclose


def test_tlwe_noiseless_trivial(thread):

    shape = (2, 5)
    params = NuFHEParameters().tgsw_params.tlwe_params

    mask_size = params.mask_size
    polynomial_degree = params.polynomial_degree

    mu = get_test_array(shape + (polynomial_degree,), Torus32)
    a = numpy.empty(shape + (mask_size + 1, polynomial_degree), Torus32)
    cv = numpy.empty(shape, ErrorFloat)

    a_dev = thread.empty_like(a)
    cv_dev = thread.empty_like(cv)
    mu_dev = thread.to_device(mu)

    test = TLweNoiselessTrivial(params, shape).compile(thread)
    ref = TLweNoiselessTrivialReference(params, shape)

    test(a_dev, cv_dev, mu_dev)
    a_test = a_dev.get()
    cv_test = cv_dev.get()

    ref(a, cv, mu)

    assert (a_test == a).all()
    assert errors_allclose(cv_test, cv)


def test_tlwe_extract_lwe_samples(thread):

    shape = (2, 5)
    params = NuFHEParameters().tgsw_params.tlwe_params

    mask_size = params.mask_size
    polynomial_degree = params.polynomial_degree

    tlwe_a = get_test_array(shape + (mask_size + 1, polynomial_degree), Torus32)

    a = numpy.empty(shape + (params.extracted_lweparams.size,), Torus32)
    b = numpy.empty(shape, Torus32)

    tlwe_a_dev = thread.to_device(tlwe_a)
    a_dev = thread.empty_like(a)
    b_dev = thread.empty_like(b)

    test = TLweExtractLweSamples(params, shape).compile(thread)
    ref = TLweExtractLweSamplesReference(params, shape)

    test(a_dev, b_dev, tlwe_a_dev)
    a_test = a_dev.get()
    b_test = b_dev.get()

    ref(a, b, tlwe_a)

    assert (a_test == a).all()
    assert (b_test == b).all()


def test_tlwe_encrypt_zero(thread):

    nufhe_params = NuFHEParameters()
    perf_params = PerformanceParameters(nufhe_params).for_device(thread.device_params)
    params = nufhe_params.tgsw_params.tlwe_params

    mask_size = params.mask_size
    polynomial_degree = params.polynomial_degree
    noise = params.min_noise

    shape = (3, 4, 5)

    result_a = numpy.empty(shape + (mask_size + 1, polynomial_degree), Torus32)
    result_cv = numpy.empty(shape, ErrorFloat)
    noises1 = get_test_array(shape + (mask_size, polynomial_degree), Torus32)
    noises2 = get_test_array(shape + (polynomial_degree,), Torus32)
    key = get_test_array((mask_size, polynomial_degree), Int32, (0, 2))

    test = TLweEncryptZero(params, shape, noise, perf_params).compile(thread)
    ref = TLweEncryptZeroReference(params, shape, noise, perf_params)

    result_a_dev = thread.empty_like(result_a)
    result_cv_dev = thread.empty_like(result_cv)
    noises1_dev = thread.to_device(noises1)
    noises2_dev = thread.to_device(noises2)
    key_dev = thread.to_device(key)

    test(result_a_dev, result_cv_dev, key_dev, noises1_dev, noises2_dev)
    ref(result_a, result_cv, key, noises1, noises2)

    result_a_test = result_a_dev.get()
    result_cv_test = result_cv_dev.get()

    assert (result_a_test == result_a).all()
    assert errors_allclose(result_cv_test, result_cv)
