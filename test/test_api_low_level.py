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

import io
import random

import numpy
import pytest

from nufhe import *


@pytest.fixture(scope='module', params=[False, True], ids=['to_file', 'to_string'])
def to_file(request):
    return request.param


def test_serialize_secret_key(thread, key_pair, to_file):
    secret_key, cloud_key = key_pair

    if to_file:
        file_obj = io.BytesIO()
        secret_key.dump(file_obj)
        file_obj.seek(0)
        secret_key_loaded = NuFHESecretKey.load(file_obj, thread)
    else:
        s = secret_key.dumps()
        secret_key_loaded = NuFHESecretKey.loads(s, thread)

    assert secret_key_loaded == secret_key


def test_serialize_cloud_key(thread, key_pair, to_file):
    secret_key, cloud_key = key_pair

    if to_file:
        file_obj = io.BytesIO()
        cloud_key.dump(file_obj)
        file_obj.seek(0)
        cloud_key_loaded = NuFHECloudKey.load(file_obj, thread)
    else:
        s = cloud_key.dumps()
        cloud_key_loaded = NuFHECloudKey.loads(s, thread)

    assert cloud_key_loaded == cloud_key


def test_serialize_ciphertext(thread, key_pair, to_file):

    secret_key, cloud_key = key_pair
    size = 32
    rng = DeterministicRNG()
    bits = [random.choice([False, True]) for i in range(size)]
    ciphertext = encrypt(thread, rng, secret_key, bits)

    if to_file:
        file_obj = io.BytesIO()
        ciphertext.dump(file_obj)
        file_obj.seek(0)
        ciphertext_loaded = LweSampleArray.load(file_obj, thread)
    else:
        s = ciphertext.dumps()
        ciphertext_loaded = LweSampleArray.loads(s, thread)

    assert ciphertext_loaded == ciphertext
