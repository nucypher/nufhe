# Copyright (C) 2018 NuCypher
#
# This file is part of nufhe.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy

from nufhe.numeric_functions import Torus32, Int32
from nufhe.numeric_functions_gpu import Torus32ToPhase
from nufhe.numeric_functions_cpu import Torus32ToPhaseReference

from utils import get_test_array, errors_allclose


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

    assert errors_allclose(result_test, result)
