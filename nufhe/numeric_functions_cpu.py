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
