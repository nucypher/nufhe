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

import numpy
import random

from os import urandom
from reikna.helpers import min_blocks

from .numeric_functions import double_to_t32, Torus32, Int32


# Used to generate uniform floats.
Float = numpy.dtype('float64')
MantissaInt = numpy.dtype('uint64') # an integer type that fits all possible mantissa values

BPF = numpy.finfo(Float).nmant + 1 # mantissa size + the implicit constant bit
RECIP_BPF = 2**(-BPF)


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
        nbytes = min_blocks(length, 8)
        random_bytes = numpy.frombuffer(urandom(nbytes), numpy.uint8)
        random_bits = numpy.unpackbits(random_bytes)[:length]
        return random_bits.reshape(shape).astype(Int32)

    def uniform_torus32(self, shape):
        length = numpy.prod(shape)
        nbytes = length * numpy.dtype(Int32).itemsize
        return numpy.frombuffer(urandom(nbytes), Int32).reshape(shape)

    def _uniform_float(self, length):
        """
        Returns an array of uniformly distributed floats in the interval (0, 1)
        (open at both ends!).
        """

        # The number of possible values in the interval [0, 1) is
        # 2**(mantissa size + the implicit constant bit)
        # (and they are uniformly distributed).

        nbytes = length * MantissaInt.itemsize
        mantissa_bits = numpy.frombuffer(urandom(nbytes), MantissaInt)

        # We want an open interval, so we want to exclude 0 without using rejection sampling.
        # To do that, we drop an additional bit from our generated integer,
        # and shift the range from [0, 2^(bpf-1)-1] to [2^(-bpf), 1 - 2^(-bpf)]
        mantissa_bits = mantissa_bits >> (MantissaInt.itemsize * 8 - (BPF - 1))
        mantissa_bits = mantissa_bits * 2 + 1

        return mantissa_bits * RECIP_BPF

    def gauss(self, shape, std_dev):
        orig_length = numpy.prod(shape)

        # Make the length even since Box-Muller transform generates pairs of randoms
        length = orig_length + orig_length % 2

        # Box-Muller transform

        u1 = self._uniform_float(length // 2)
        u2 = self._uniform_float(length // 2)

        r = (-2 * numpy.log(u1))**0.5
        theta = 2 * numpy.pi * u2

        z0 = r * numpy.cos(theta)
        z1 = r * numpy.sin(theta)

        result = numpy.concatenate([z0, z1])[:orig_length]

        return result.reshape(shape) * std_dev


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
