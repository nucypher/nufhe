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


class GaloisNumber:

    modulus = 2**64 - 2**32 + 1
    factors = [2, 3, 5, 17, 257, 65537] # prime factors of (modulus - 1)

    def __init__(self, val):
        self.val = int(val) % self.modulus

    def __add__(self, other: "GaloisNumber"):
        return GaloisNumber((self.val + other.val) % self.modulus)

    def __sub__(self, other: "GaloisNumber"):
        return GaloisNumber((self.val - other.val) % self.modulus)

    def __mul__(self, other: "GaloisNumber"):
        return GaloisNumber((self.val * other.val) % self.modulus)

    def __truediv__(self, other: "GaloisNumber"):
        return self * other.inverse()

    def __pow__(self, e: int):
        if e == 0:
            return GaloisNumber(1)
        elif e == 1:
            return self
        else:
            x = GaloisNumber(self.val)
            y = GaloisNumber(1)
            while e > 1:
                if e % 2 != 0:
                    y *= x
                x *= x
                e //= 2
            return x * y

    def inverse(self):
        return self**(self.modulus - 2)

    def __eq__(self, other):
        if isinstance(other, GaloisNumber):
            return self.val == other.val
        else:
            return self.val == other

    def __repr__(self):
        return "GaloisNumber(" + str(self.val) + ")"

    def __str__(self):
        return str(self.val) + "G"


gnum = numpy.vectorize(GaloisNumber)

def _gnum_to_i32(x):
    # Treats any value less than the half of the finite field modulus as a positive integer,
    # and anything greater that that as a negative integer -(modulus - x),
    # then truncate the result to the i32 range.
    med = x.modulus // 2
    val = x.val
    return numpy.int32(val & 0xffffffff) - (val > med)

gnum_to_i32 = numpy.vectorize(_gnum_to_i32)

gnum_to_u64 = numpy.vectorize(lambda x: numpy.uint64(x.val))


def find_generator(start=2):
    for w in range(start, GaloisNumber.modulus):
        w = GaloisNumber(w)
        for q in GaloisNumber.factors:
            if w**((GaloisNumber.modulus - 1) // q) == 1:
                break
        else:
            return w


def root_of_unity(N):
    """
    Returns a root of unity of order N (that is, x^N=1).
    Different roots produce different results.
    A standard approach is to use

        find_generator()**((GaloisNumber.modulus - 1) // N)

    To compare the results against the GPU version,
    we return the root the GPU implementation uses.
    """
    assert 2**32 % N == 0
    return GaloisNumber(0xa70dc47e4cbdf43f)**(2**32 // N)


def galois_fft_base(N, inverse):
    w = root_of_unity(N)
    if inverse:
        return GaloisNumber(1) / w
    else:
        return w


def galois_fft_inverse_coeff(N):
    return GaloisNumber(1) / GaloisNumber(N)


def ntt_naive(a, inverse):
    N = a.size
    w = galois_fft_base(N, inverse)

    result = gnum(numpy.zeros(N))
    for i in range(N):
        for j in range(N):
            result[i] += a[j] * w**(i * j)

    if inverse:
        return result * galois_fft_inverse_coeff(N)
    else:
        return result


def bitreverse(x, l):
    # Slow, but simple function. Needs some optimized algorithm in practice.
    b = ('{:0' + str(l) + 'b}').format(x)
    return int(b[::-1], 2)


def fft_generic(fft_base_func, fft_inverse_coeff_func, data, inverse):

    batch_shape, n = data.shape[:-1], data.shape[-1]
    data = data.reshape(numpy.prod((1,) + batch_shape), n).transpose().copy()

    logn = numpy.round(numpy.log2(n)).astype(numpy.int32)

    assert n >= 2
    assert n == 2**logn

    for i in range(n):
        j = bitreverse(i, logn)
        if j > i:
            data[[j, i]] = data[[i, j]]

    w = fft_base_func(n, inverse)

    for stage in range(logn):
        mmax = 2**stage
        istep = mmax * 2

        for m in range(mmax):
            tw = w**(m * 2**(logn - stage - 1))

            for i in range(m, n, istep):
                j = i + mmax

                temp = data[j] * tw
                data[j] = data[i] - temp
                data[i] += temp

    data = data.transpose().reshape(batch_shape + (n,))

    if inverse:
        return data * fft_inverse_coeff_func(n)
    else:
        return data


def ntt(data, inverse):
    return fft_generic(galois_fft_base, galois_fft_inverse_coeff, data, inverse)
