import numpy

from reikna.cluda import Module


Torus32 = numpy.int32
Float = numpy.float64


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


dtot32_gpu = Module.create(
    """
    WITHIN_KERNEL int ${prefix}(double d)
    {
        return (d - trunc(d)) * ${2**32};
    }
    """)


def int64_to_int32(x: int):
    return x.astype(numpy.int32)


def float_to_int32(x: float):
    return numpy.round(x).astype(numpy.int64).astype(numpy.int32)
