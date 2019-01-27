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

import random
import io

import numpy
import pytest

import nufhe


@pytest.mark.parametrize('make_pair', [False, True], ids=["make keys separately", "make_key_pair"])
def test_context(make_pair):

    size = 32
    bits1 = [random.choice([False, True]) for i in range(size)]
    bits2 = [random.choice([False, True]) for i in range(size)]
    reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]

    ctx = nufhe.Context()

    if make_pair:
        secret_key, cloud_key = ctx.make_key_pair()
    else:
        secret_key = ctx.make_secret_key()
        cloud_key = ctx.make_cloud_key(secret_key)

    ciphertext1 = ctx.encrypt(secret_key, bits1)
    ciphertext2 = ctx.encrypt(secret_key, bits2)

    vm = ctx.make_virtual_machine(cloud_key)
    result = vm.gate_nand(ciphertext1, ciphertext2)
    result_bits = ctx.decrypt(secret_key, result)

    assert all(result_bits == reference)


@pytest.fixture(scope='module', params=[False, True], ids=['to_file', 'to_string'])
def to_file(request):
    return request.param


def test_serialize_secret_key(to_file):

    ctx = nufhe.Context()
    secret_key = ctx.make_secret_key()

    if to_file:
        file_obj = io.BytesIO()
        secret_key.dump(file_obj)
        file_obj.seek(0)
        secret_key_loaded = ctx.load_secret_key(file_obj)
    else:
        s = secret_key.dumps()
        secret_key_loaded = ctx.load_secret_key(s)

    assert secret_key_loaded == secret_key


def test_serialize_cloud_key(to_file):

    ctx = nufhe.Context()
    secret_key, cloud_key = ctx.make_key_pair()

    if to_file:
        file_obj = io.BytesIO()
        cloud_key.dump(file_obj)
        file_obj.seek(0)
        cloud_key_loaded = ctx.load_cloud_key(file_obj)
    else:
        s = cloud_key.dumps()
        cloud_key_loaded = ctx.load_cloud_key(s)

    assert cloud_key_loaded == cloud_key


def test_serialize_ciphertext(to_file):

    ctx = nufhe.Context()
    secret_key, cloud_key = ctx.make_key_pair()

    size = 32
    bits = [random.choice([False, True]) for i in range(size)]
    ciphertext = ctx.encrypt(secret_key, bits)

    if to_file:
        file_obj = io.BytesIO()
        ciphertext.dump(file_obj)
        file_obj.seek(0)
        ciphertext_loaded = ctx.load_ciphertext(file_obj)
    else:
        s = ciphertext.dumps()
        ciphertext_loaded = ctx.load_ciphertext(s)

    assert ciphertext_loaded == ciphertext


@pytest.mark.parametrize('rng_cls', [nufhe.DeterministicRNG, nufhe.SecureRNG])
def test_rngs(rng_cls):

    size = 32
    bits1 = [random.choice([False, True]) for i in range(size)]
    bits2 = [random.choice([False, True]) for i in range(size)]
    reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]

    rng = rng_cls()
    ctx = nufhe.Context(rng=rng)
    secret_key, cloud_key = ctx.make_key_pair()

    ciphertext1 = ctx.encrypt(secret_key, bits1)
    ciphertext2 = ctx.encrypt(secret_key, bits2)

    vm = ctx.make_virtual_machine(cloud_key)
    result = vm.gate_nand(ciphertext1, ciphertext2)
    result_bits = ctx.decrypt(secret_key, result)

    assert all(result_bits == reference)
