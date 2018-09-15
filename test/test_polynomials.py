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

import time

import pytest
import numpy

from nufhe.numeric_functions import Torus32, Int32
from nufhe.polynomials_gpu import ShiftTorusPolynomial
from nufhe.polynomials_cpu import ShiftTorusPolynomialReference

from utils import get_test_array, errors_allclose


@pytest.mark.parametrize('option', ['minus_one', 'invert_powers', 'powers_view'])
def test_shift_torus_polynomial(thread, option):

    polynomial_degree = 16
    shape = (20, 30)

    powers_shape = (20, 10) if option == 'powers_view' else (20,)
    powers_idx = 5 # not used unless `option == 'powers_view'`

    source = get_test_array(shape + (polynomial_degree,), Torus32)
    powers = get_test_array(powers_shape, Int32, (0, 2 * polynomial_degree))

    result = numpy.empty_like(source)

    source_dev = thread.to_device(source)
    powers_dev = thread.to_device(powers)
    result_dev = thread.empty_like(result)

    options = {option: True}

    comp = ShiftTorusPolynomial(polynomial_degree, shape, powers_shape, **options).compile(thread)
    ref = ShiftTorusPolynomialReference(polynomial_degree, shape, powers_shape, **options)

    comp(result_dev, source_dev, powers_dev, powers_idx)
    result_test = result_dev.get()

    ref(result, source, powers, powers_idx)

    assert errors_allclose(result_test, result)
