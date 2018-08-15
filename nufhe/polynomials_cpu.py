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

from reikna.helpers import product


# minus_one=True: result = (X^ai-1) * source
# minus_one=False: result = X^{a} * source
def ShiftTorusPolynomialReference(
        polynomial_degree, shape, powers_shape,
        powers_view=False, minus_one=False, invert_powers=False):

    batch_shape = powers_shape[:-1] if powers_view else powers_shape
    assert batch_shape == shape[:len(batch_shape)]
    poly_batch_shape = shape[len(batch_shape):]

    def _kernel(result, source, powers, powers_idx):

        if powers_view:
            powers = powers.reshape(product(batch_shape), powers_shape[-1])[:, powers_idx]
        else:
            powers = powers.flatten()

        result = result.reshape(product(batch_shape), product(poly_batch_shape), polynomial_degree)
        source = source.reshape(product(batch_shape), product(poly_batch_shape), polynomial_degree)

        if invert_powers:
            powers = 2 * polynomial_degree - powers

        for i in range(result.shape[0]):
            power = powers[i]
            if power < polynomial_degree:
                result[i,:,:power] = -source[i,:,(polynomial_degree - power):polynomial_degree]
                result[i,:,power:polynomial_degree] = source[i,:,:(polynomial_degree - power)]
            else:
                power = power - polynomial_degree
                result[i,:,:power] = source[i,:,(polynomial_degree - power):polynomial_degree]
                result[i,:,power:polynomial_degree] = -source[i,:,:(polynomial_degree - power)]

        if minus_one:
            result -= source

    return _kernel
