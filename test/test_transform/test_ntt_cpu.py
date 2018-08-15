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

from nufhe.transform.ntt_cpu import (
    GaloisNumber, gnum, ntt_naive, ntt, find_generator, gnum_to_i32)


def test_ntt():
    # Compare the naive NTT (O(N^2) complexity)
    # with the NTT based on the FFT scheme (O(N log N) complexity)

    a = gnum(numpy.random.randint(0, 1000, size=16))

    af = ntt(a, False)
    ab = ntt(af, True)

    af_ref = ntt_naive(a, False)
    ab_ref = ntt_naive(af_ref, True)

    assert (a == ab).all()
    assert (af == af_ref).all()
    assert (ab == ab_ref).all()


def test_find_generator():
    # Check that the returned generator

    g = find_generator(start=2)
    modulus = GaloisNumber.modulus

    for q in GaloisNumber.factors:
        assert g**((modulus - 1) // q) != 1

    assert g**(modulus - 1) == 1


def test_repr():
    # A serialization test
    a = GaloisNumber(1000)
    assert repr(a) == "GaloisNumber(1000)"
    assert str(a) == "1000G"


def test_gnum_to_i32():
    modulus = GaloisNumber.modulus
    assert gnum_to_i32(GaloisNumber(12345)) == 12345
    assert gnum_to_i32(GaloisNumber(2**31 + 12345)) == -2**31 + 12345
    assert gnum_to_i32(GaloisNumber(2**32 + 12345)) == 12345
    assert gnum_to_i32(GaloisNumber(modulus - 12345)) == -12345
    assert gnum_to_i32(GaloisNumber(modulus - (2**31 + 12345))) == 2**31 - 12345
    assert gnum_to_i32(GaloisNumber(modulus - (2**32 + 12345))) == -12345

