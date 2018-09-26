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

import pickle

import numpy

from .utils import arrays_equal
from .computation_cache import get_computation
from .numeric_functions import Torus32, Int32
from .polynomial_transform import get_transform
from .polynomials_gpu import ShiftTorusPolynomial


# This structure represents an integer polynomial modulo X^N+1
class IntPolynomialArray:

    def __init__(self, coeffs):
        assert coeffs.dtype == Int32
        self.coeffs = coeffs
        self.shape = coeffs.shape[:-1]
        self.polynomial_degree = coeffs.shape[-1]


# This structure represents an torus polynomial modulo X^N+1
class TorusPolynomialArray:

    def __init__(self, coeffs):
        assert coeffs.dtype == Torus32
        self.coeffs = coeffs
        self.shape = coeffs.shape[:-1]
        self.polynomial_degree = coeffs.shape[-1]

    @classmethod
    def empty(cls, thr, polynomial_degree: int, shape):
        return cls(thr.array(shape + (polynomial_degree,), Torus32))


# Torus polynomial in transformed space
class TransformedPolynomialArray:

    def __init__(self, transform_type, coeffs):
        transform = get_transform(transform_type)
        assert coeffs.dtype == transform.transformed_dtype()
        self.transform_type = transform_type
        self.coeffs = coeffs
        self.shape = coeffs.shape[:-1]
        self.polynomial_degree = coeffs.shape[-1]

    @classmethod
    def empty(cls, thr, transform_type, polynomial_degree: int, shape):
        transform = get_transform(transform_type)
        coeffs = thr.array(
            shape + (transform.transformed_length(polynomial_degree),),
            transform.transformed_dtype())
        return cls(transform_type, coeffs)

    def dump(self, file_obj):
        pickle.dump(self.transform_type, file_obj)
        pickle.dump(self.coeffs.get(), file_obj)

    @classmethod
    def load(cls, file_obj, thr):
        transform_type = pickle.load(file_obj)
        coeffs = pickle.load(file_obj)
        return cls(transform_type, thr.to_device(coeffs))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.transform_type == other.transform_type
            and arrays_equal(self.coeffs, other.coeffs))


# result = X^(2N - pwr) * source
def shift_tp_inverted_power(
        thr, result: TorusPolynomialArray, powers, source: TorusPolynomialArray):
    comp = get_computation(
        thr, ShiftTorusPolynomial,
        result.polynomial_degree, result.shape, powers.shape, invert_powers=True)
    comp(result.coeffs, source.coeffs, powers, 0)


# result = (X^pwr - 1) * source
def shift_tp_minus_one_power_from_array(
        thr, result: TorusPolynomialArray, powers, power_idx: int, source: TorusPolynomialArray):
    comp = get_computation(
        thr, ShiftTorusPolynomial,
        result.polynomial_degree, result.shape, powers.shape, powers_view=True, minus_one=True)
    comp(result.coeffs, source.coeffs, powers, power_idx)
