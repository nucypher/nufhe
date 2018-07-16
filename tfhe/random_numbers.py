"""
At the moment the random numer generating functions simply use a CPU RNG
and transfer the result to GPU. The reasons are:

- random number generation is fast compared to the rest of the algorithms
- a GPU RNG requires some non-trivial logistics with preserving/creating the random state
- it makes it is easier to test the rest of the code during the transition from CPU to GPU

When it is necessary, the functions can be made to execute on GPU without changing the API.
"""

import numpy

from .numeric_functions import dtot32, Torus32


def _rand_uniform_int32(rng, shape):
    return rng.randint(0, 2, size=shape, dtype=numpy.int32)


def _rand_uniform_torus32(rng, shape):
    return rng.randint(-2**31, 2**31, size=shape, dtype=Torus32)


def _rand_gaussian_float(rng, sigma: float, shape):
    return rng.normal(size=shape, scale=sigma)


# Gaussian sample centered in message, with standard deviation sigma
def _rand_gaussian_torus32(rng, message: Torus32, sigma: float, shape):
    # Attention: all the implementation will use the stdev instead of the gaussian fourier param
    return message + dtot32(rng.normal(size=shape, scale=sigma))


def rand_uniform_int32(thr, rng, shape):
    return thr.to_device(_rand_uniform_int32(rng, shape))


def rand_uniform_torus32(thr, rng, shape):
    return thr.to_device(_rand_uniform_torus32(rng, shape))


def rand_gaussian_float(thr, rng, sigma, shape):
    return thr.to_device(_rand_gaussian_float(rng, sigma, shape))

def rand_gaussian_torus32(thr, rng, message: Torus32, sigma: float, shape):
    return thr.to_device(_rand_gaussian_torus32(rng, message, sigma, shape))
