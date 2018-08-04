import numpy

from .numeric_functions import Torus32


def modSwitchFromTorus32_reference(res, phase, Msize: int):

    assert phase.dtype == Torus32
    assert res.dtype == numpy.int32
    assert phase.shape == res.shape

    interv = numpy.uint32(2**32 // Msize)
    half_interv = numpy.uint32(interv // 2)
    numpy.copyto(res, ((phase.astype(numpy.uint32) + half_interv) // interv).astype(numpy.int32))
