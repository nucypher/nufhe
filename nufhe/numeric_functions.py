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

from reikna.cluda import Module

from .numeric_functions_gpu import Torus32ToPhase
from .numeric_functions_gpu import Torus32, Int32, ErrorFloat # for re-export
from .computation_cache import get_computation


# Approximate the phase to the nearest message possible in the message space.
# The constant `mspace_size` indicates on which message space we are working
# (how many messages possible).
def phase_to_t32(phase: int, mspace_size: int):
    return Torus32((phase % mspace_size) * (2**32 // mspace_size))


def t32_to_phase(thr, result, messages, mspace_size: int):
    comp = get_computation(thr, Torus32ToPhase, messages.shape, mspace_size)
    comp(result, messages)


def double_to_t32(d: float):
    return ((d - numpy.trunc(d)) * 2**32).astype(Torus32)
