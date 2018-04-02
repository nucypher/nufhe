import numpy


Torus32 = numpy.int32


def rand_uniform_int32(rng, shape):
    return rng.randint(0, 2, size=shape, dtype=numpy.int32)


def rand_uniform_torus32(rng, shape):
    # TODO: if dims == () (it happens), the return value is not an array -> type instability
    #       also, there's probably instability for arrays of different dims too.
    #       Naturally, it applies for all other rand_ functions.
    return rng.randint(-2**31, 2**31, size=shape, dtype=Torus32)


def rand_gaussian_float(rng, sigma: float, shape):
    return rng.normal(size=shape, scale=sigma)


# Gaussian sample centered in message, with standard deviation sigma
def rand_gaussian_torus32(rng, message: Torus32, sigma: float, shape):
    # Attention: all the implementation will use the stdev instead of the gaussian fourier param
    return message + dtot32(rng.normal(size=shape, scale=sigma))


# Used to approximate the phase to the nearest message possible in the message space
# The constant Msize will indicate on which message space we are working (how many messages possible)
#
# "work on 63 bits instead of 64, because in our practical cases, it's more precise"
def modSwitchFromTorus32(phase: Torus32, Msize: int):
    # TODO: check if it can be simplified (wrt type conversions)
    interv = (1 << 63) // Msize * 2 # width of each intervall
    half_interval = interv // 2 # begin of the first intervall
    phase64 = (phase.astype(numpy.uint32).astype(numpy.uint64) << 32) + half_interval
    # floor to the nearest multiples of interv
    return (phase64 // interv).astype(numpy.int64).astype(numpy.int32)


# Used to approximate the phase to the nearest message possible in the message space
# The constant Msize will indicate on which message space we are working (how many messages possible)
#
# "work on 63 bits instead of 64, because in our practical cases, it's more precise"
def modSwitchToTorus32(mu: int, Msize: int):

    interv = ((1 << 63) // Msize) * 2 # width of each intervall
    phase64 = mu * interv
    # floor to the nearest multiples of interv
    return Torus32(phase64 >> 32)


# from double to Torus32
def dtot32(d: float):
    return ((d - numpy.trunc(d)) * 2**32).astype(numpy.int32)


def int64_to_int32(x: int):
    return x.astype(numpy.int32)


def float_to_int32(x: float):
    return numpy.round(x).astype(numpy.int64).astype(numpy.int32)
