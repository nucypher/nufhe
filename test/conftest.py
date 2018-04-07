import pytest


@pytest.fixture(scope='module')
def thread():
    from reikna.cluda import ocl_api, cuda_api
    api = cuda_api()
    return api.Thread.create()
