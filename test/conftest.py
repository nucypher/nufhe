import pytest

from reikna.cluda import cuda_api, ocl_api

from tfhe.computation_cache import clear_computation_cache


@pytest.fixture(scope='session', params=["cuda"])
def thread(request):
    if request.param == 'cuda':
        api = cuda_api()
        platform = api.get_platforms()[0]
        device = platform.get_devices()[0]
    else:
        api = ocl_api()
        platform = api.get_platforms()[0]
        device = platform.get_devices()[2]

    thread = api.Thread(device)
    yield thread

    # Computations may retain references to the Thread objects,
    # so we need to clear the cache first so that the thread could be destroyed
    # as it goes out of scope.
    # CUDA is sensitive to the exact timing of the destruction
    # because of the stateful nature of its API.
    clear_computation_cache()
