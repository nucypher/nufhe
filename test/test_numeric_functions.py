import numpy

from tfhe.numeric_functions import Torus32
from tfhe.gpu_numeric_functions import ModSwitchFromTorus32


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


def test_modSwitchFromTorus32(thread):

    Msize = 2048
    phase = numpy.random.randint(-2**31, 2**31, size=(10, 20, 30), dtype=Torus32)
    res_ref = numpy.empty(phase.shape, numpy.int32)

    phase_dev = thread.to_device(phase)
    res_dev = thread.empty_like(res_ref)

    comp = ModSwitchFromTorus32(phase).compile(thread)

    comp(res_dev, phase_dev, Msize)
    res_test = res_dev.get()

    modSwitchFromTorus32_reference(res_ref, phase, Msize)

    assert numpy.allclose(res_test, res_ref)
