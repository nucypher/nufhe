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

from .api_low_level import (
    make_key_pair,
    encrypt,
    decrypt,
    empty_ciphertext,
    NuFHEParameters,
    NuFHESecretKey,
    NuFHECloudKey,
    )

from .lwe import (
    LweSampleArray,
    concatenate,
    )

from .gates import (
    gate_nand,
    gate_or,
    gate_and,
    gate_xor,
    gate_xnor,
    gate_not,
    gate_copy,
    gate_constant,
    gate_nor,
    gate_andny,
    gate_andyn,
    gate_orny,
    gate_oryn,
    gate_mux,
    )

from .performance import PerformanceParameters

from .random_numbers import DeterministicRNG, SecureRNG

from .computation_cache import clear_computation_cache

from .api_high_level import (
    find_devices,
    Context,
    )
