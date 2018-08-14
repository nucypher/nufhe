import numpy

from .numeric_functions_gpu import Torus32, Int32


def Torus32ToPhaseReference(shape, mspace_size: int):

    interv = numpy.uint32(2**32 // mspace_size)
    half_interv = numpy.uint32(interv // 2)

    def _kernel(result, phase):

        nonlocal interv

        assert phase.dtype == Torus32
        assert result.dtype == Int32

        numpy.copyto(result, ((phase.astype(numpy.uint32) + half_interv) // interv).astype(Int32))

    return _kernel
