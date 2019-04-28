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
import pytest

from reikna.cluda import cuda_api, ocl_api, get_api, supported_api_ids, find_devices

from nufhe import make_key_pair, DeterministicRNG, Context
from nufhe.computation_cache import clear_computation_cache


def pytest_addoption(parser):
    api_ids = supported_api_ids()

    parser.addoption("--api", action="store",
        help="Backend API. If 'supported' is chosen, will run on all available ones.",
        default="supported", choices=api_ids + ["supported"])
    parser.addoption("--device-include-mask", action="append",
        help="Run tests on matching devices only",
        default=[])
    parser.addoption("--include-duplicate-devices", action="store_true",
        help="Run tests on all available devices and not only on uniquely named ones.",
        default=False)
    parser.addoption("--heavy-performance-load", action="store_true",
        help=(
            "Use large data sizes and numbers of iterations for performance tests. "
            "Recommended for high-tier videocards."),
        default=False)
    parser.addoption("--transform", action="store",
        help=(
            "The type of polynomial transform to use for tests "
            "that can use different transform types."),
        default="all", choices=["NTT", "FFT", "all"])


def pytest_generate_tests(metafunc):

    config = metafunc.config

    if 'thread' in metafunc.fixturenames:

        api_opt = config.option.api
        api_ids = supported_api_ids() if api_opt == 'supported' else [api_opt]

        vals = []
        ids = []
        for api_id in api_ids:
            api = get_api(api_id)
            devices = find_devices(
                api,
                include_devices=config.option.device_include_mask,
                include_duplicate_devices=config.option.include_duplicate_devices)
            for pnum in sorted(devices.keys()):
                dnums = sorted(devices[pnum])
                for dnum in dnums:
                    vals.append((api_id, pnum, dnum))
                    ids.append("{api_id}:{pnum}:{dnum}".format(api_id=api_id, pnum=pnum, dnum=dnum))

        if len(vals) == 0:
            raise RuntimeError(
                "Neither PyCUDA nor PyOpenCL could find any suitable devices. "
                "Check your system configuration.")

        metafunc.parametrize("thread", vals, ids=ids, indirect=True)

    if 'transform_type' in metafunc.fixturenames:

        if config.option.transform == 'all':
            vals = ['NTT', 'FFT']
        else:
            vals = [config.option.transform]

        metafunc.parametrize("transform_type", vals)


@pytest.fixture(scope='session')
def thread(request):

    api_id, pnum, dnum = request.param

    api = get_api(api_id)

    platform = api.get_platforms()[pnum]
    device = platform.get_devices()[dnum]

    thread = api.Thread(device)
    yield thread

    # Computations may retain references to the Thread objects,
    # so we need to clear the cache first so that the thread could be destroyed
    # as it goes out of scope.
    # CUDA is sensitive to the exact timing of the destruction
    # because of the stateful nature of its API.
    clear_computation_cache(thread)


@pytest.fixture(scope='session')
def heavy_performance_load(request):
    return request.config.option.heavy_performance_load


@pytest.fixture(scope='session')
def key_pair(thread):
    rng = DeterministicRNG()
    secret_key, cloud_key = make_key_pair(thread, rng)
    return secret_key, cloud_key


@pytest.fixture(scope='session')
def context(thread):
    return Context(thread=thread)


@pytest.fixture(scope='session')
def context_and_key_pair(context):
    secret_key, cloud_key = context.make_key_pair()
    return context, secret_key, cloud_key
