import numpy

from .numeric_functions import Torus32


def modSwitchFromTorus32_reference(res, phase, Msize: int):

    assert phase.dtype == Torus32
    assert res.dtype == numpy.int32
    assert phase.shape == res.shape

    # TODO: check if it can be simplified (wrt type conversions)
    interv = (1 << 63) // Msize * 2 # width of each intervall
    half_interval = interv // 2 # begin of the first intervall
    phase64 = (phase.astype(numpy.uint32).astype(numpy.uint64) << 32) + half_interval
    # floor to the nearest multiples of interv
    numpy.copyto(res, (phase64 // interv).astype(numpy.int64).astype(numpy.int32))
