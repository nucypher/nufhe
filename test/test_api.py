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

import numpy

from nufhe import *


def test_serialize_secret_key(thread, key_pair):
    secret_key, cloud_key = key_pair

    file_obj = io.BytesIO()
    secret_key.dump(file_obj)

    file_obj.seek(0)
    secret_key_loaded = NuFHESecretKey.load(file_obj, thread)

    assert secret_key_loaded == secret_key


def test_serialize_cloud_key(thread, key_pair):
    secret_key, cloud_key = key_pair

    file_obj = io.BytesIO()
    cloud_key.dump(file_obj)

    file_obj.seek(0)
    cloud_key_loaded = NuFHECloudKey.load(file_obj, thread)

    assert cloud_key_loaded == cloud_key


def test_serialize_ciphertext(thread, key_pair):

    secret_key, cloud_key = key_pair
    size = 32
    rng = numpy.random.RandomState()
    bits = rng.randint(0, 2, size=size).astype(numpy.bool)
    ciphertext = encrypt(thread, rng, secret_key, bits)

    file_obj = io.BytesIO()
    ciphertext.dump(file_obj)

    file_obj.seek(0)
    ciphertext_loaded = LweSampleArray.load(file_obj, thread)

    assert ciphertext_loaded == ciphertext
