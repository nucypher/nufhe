import pytest

from reikna.cluda import cuda_api, ocl_api, get_api, supported_api_ids, find_devices

from tfhe.computation_cache import clear_computation_cache


def pytest_addoption(parser):
    api_ids = supported_api_ids()

    parser.addoption("--api", action="store",
        help="Backend API. If 'supported' is chosen, will run on all available ones.",
        default="supported", choices=api_ids + ["supported"])
    parser.addoption("--device-include-mask", action="append",
        help="Run tests on matching devices only",
        default=[])
    parser.addoption("--include-duplicate-devices", action="store_true",
        help="Run tests on all available devices and not only on uniquely named ones",
        default=False)


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

        metafunc.parametrize("thread", vals, ids=ids, indirect=True)


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
    clear_computation_cache()
