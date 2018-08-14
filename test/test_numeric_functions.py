import numpy

from nufhe.numeric_functions import Torus32, Int32
from nufhe.numeric_functions_gpu import Torus32ToPhase
from nufhe.numeric_functions_cpu import Torus32ToPhaseReference

from utils import get_test_array


def test_t32_to_phase(thread):

    mspace_size = 2048
    shape = (10, 20, 30)
    phase = get_test_array(shape, Torus32)
    result = numpy.empty(shape, Int32)

    phase_dev = thread.to_device(phase)
    result_dev = thread.empty_like(result)

    comp = Torus32ToPhase(shape, mspace_size).compile(thread)
    ref = Torus32ToPhaseReference(shape, mspace_size)

    comp(result_dev, phase_dev)
    result_test = result_dev.get()

    ref(result, phase)

    assert numpy.allclose(result_test, result)
