import numpy

from tfhe.numeric_functions import Torus32
from tfhe.numeric_functions_gpu import ModSwitchFromTorus32
from tfhe.numeric_functions_cpu import modSwitchFromTorus32_reference


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
