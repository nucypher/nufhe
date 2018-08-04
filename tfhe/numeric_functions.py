import numpy

from reikna.cluda import Module


Torus32 = numpy.int32
Int32 = numpy.int32
Float = numpy.float64


# Used to approximate the phase to the nearest message possible in the message space
# The constant Msize will indicate on which message space we are working (how many messages possible)
def modSwitchToTorus32(mu: int, Msize: int):
    return Torus32((mu % Msize) * (2**32 // Msize))


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
