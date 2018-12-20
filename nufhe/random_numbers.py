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

"""
At the moment the random numer generating functions simply use a CPU RNG
and transfer the result to GPU. The reasons are:

- random number generation is fast compared to the rest of the algorithms
- a GPU RNG requires some non-trivial logistics with preserving/creating the random state
- it makes it is easier to test the rest of the code during the transition from CPU to GPU

When it is necessary, the functions can be made to execute on GPU without changing the API.
"""

import random

import numpy

from .numeric_functions import double_to_t32, Torus32, Int32


class DeterministicRNG:
    """
    A fast, but not cryptographically secure RNG.
    Useful for testing, since it allows seeding the initial state.
    """

    def __init__(self, seed=None):
        self.rng = numpy.random.RandomState(seed)

    def uniform_bool(self, shape):
        return self.rng.randint(0, 2, size=shape, dtype=Int32)

    def uniform_torus32(self, shape):
        return self.rng.randint(-2**31, 2**31, size=shape, dtype=Torus32)

    def gauss(self, shape, std_dev):
        return self.rng.normal(size=shape, scale=std_dev)


class SecureRNG:
    """
    A cryptographically secure RNG provided by the OS.

    .. note::

        This RNG can be very slow, leading to cloud key creation times of the order of minutes.
        Encryption is not affected too much (the required amount of random numbers is much lower).
    """

    def __init__(self):
        self.rng = random.SystemRandom()

    def uniform_bool(self, shape):
        length = numpy.prod(shape)
        return numpy.array([self.rng.randrange(0, 2) for i in range(length)], Int32).reshape(shape)

    def uniform_torus32(self, shape):
        length = numpy.prod(shape)
        lo = -2**31
        hi = 2**31
        return numpy.array(
            [self.rng.randrange(lo, hi) for i in range(length)], Torus32).reshape(shape)

    def gauss(self, shape, std_dev):
        length = numpy.prod(shape)
        return numpy.array(
            [self.rng.normalvariate(0, std_dev) for i in range(length)]).reshape(shape)


# Gaussian sample centered in message, with standard deviation sigma
def _rand_gaussian_torus32(rng, message: Torus32, sigma: float, shape, centered=False):
    # Attention: all the implementation will use the stdev instead of the gaussian fourier param
    rfloats = rng.gauss(shape, sigma)
    if centered:
        rfloats -= rfloats.mean()
    return Torus32(message) + double_to_t32(rfloats)


def rand_uniform_bool(thr, rng, shape):
    return thr.to_device(rng.uniform_bool(shape))


def rand_uniform_torus32(thr, rng, shape):
    return thr.to_device(rng.uniform_torus32(shape))


def rand_gaussian_torus32(thr, rng, message: Torus32, sigma: float, shape, centered=False):
    return thr.to_device(_rand_gaussian_torus32(rng, message, sigma, shape, centered=centered))
