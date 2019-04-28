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
from reikna import cluda

from .lwe import LweSampleArray
from .api_low_level import (
    NuFHEParameters, NuFHESecretKey, NuFHECloudKey, encrypt, decrypt, empty_ciphertext)
from .performance import PerformanceParameters
from . import gates
from .random_numbers import DeterministicRNG
from .computation_cache import clear_computation_cache
from .gates import get_shape, result_shape


def _get_api_object(api):
    """
    Returns a Reikna API object based on API name (``None``, ``'CUDA'`` or ``'OpenCL'``).
    """
    api_funcs = {
        None: cluda.any_api,
        'CUDA': cluda.cuda_api,
        'OpenCL': cluda.ocl_api,
        }
    if api not in api_funcs:
        raise ValueError("Uknonwn GPGPU API identifier: " + str(api))
    return api_funcs[api]()


def find_devices(api=None,
        include_devices=None, exclude_devices=None,
        include_platforms=None, exclude_platforms=None):
    """
    Returns a list of computation device identifiers for the given API and selection criteria.
    If there are several platforms with suitable devices, only the first one will be used
    (so if you need a specific platform, use the corresponding masks).

    :param api: the GPGPU backend to use, one of ``None``, ``"CUDA"`` and ``"OpenCL"``.
        If ``None`` is given, an arbitrary available backend will be chosen.
    :param include_devices: a list of strings; only devices with one of the strings
        present in the name will be included.
    :param exclude_devices: a list of strings; devices with one of the strings
        present in the name will be excluded.
    :param include_platforms: a list of strings; only platforms with one of the strings
        present in the name will be included.
    :param exclude_platforms: a list of strings; platforms with one of the strings
        present in the name will be excluded.
    :returns: a list of :py:class:`~nufhe.api_high_level.DeviceID` objects.
    """

    api_obj = _get_api_object(api)

    platforms_and_devices = cluda.find_devices(
        api_obj,
        include_devices=include_devices,
        exclude_devices=exclude_devices,
        include_platforms=include_platforms,
        exclude_platforms=exclude_platforms)

    if len(platforms_and_devices) == 0:
        raise ValueError("No devices found satisfying the given search criteria")

    platform_ids = list(platforms_and_devices)

    return [
        DeviceID(api_obj.get_id(), platform_ids[0], device_id)
        for device_id in platforms_and_devices[platform_ids[0]]]


class DeviceID:
    """
    An identifier of a computation device suitable to run NuFHE.
    Obtained from :py:func:`~nufhe.find_devices`.
    Can be passed to another thread/process and used to create a :py:class:`~nufhe.Context` object.

    .. py:attribute:: api_name

        The name of the API (``"CUDA"`` or ``"OpenCL"``).

    .. py:attribute:: platform_name

        The name of the platform.

    .. py:attribute:: device_name

        The name of the device.
    """

    def __init__(self, api_id, platform_id, device_id):
        self.api_id = api_id
        self.platform_id = platform_id
        self.device_id = device_id

        # We do not save the actual device_obj because this object
        # can be passed to another thread/process.
        api, platform, device = self._get_objects()
        self.platform_name = platform.name
        self.device_name = device.name
        self.api_name = "CUDA" if api.get_id() == cluda.cuda_id() else "OpenCL"

    def _get_objects(self):
        api_obj = cluda.get_api(self.api_id)
        platform = api_obj.get_platforms()[self.platform_id]
        return api_obj, platform, platform.get_devices()[self.device_id]

    def get_api_and_device(self):
        api_obj, _, device = self._get_objects()
        return api_obj, device

    def __str__(self):
        return "DeviceID({api}, {platform}, {device})".format(
            api=self.api_name, platform=self.platform_name, device=self.device_name)


class Context:
    """
    An object encapuslating an execution environment on a GPU.

    If ``thread`` is given, it will be used to create the context;
    otherwise, if ``device_id`` is given, it will be used;
    if none of the above is given, the first device satisfying the given criteria will be used.

    :param rng: a random number generator which will be used wherever randomness is required.
        Can be an instance of one of the :ref:`random-number-generators`
        (:py:class:`DeterministicRNG` by default).
    :param thread: a Reikna ``Thread`` object to use internally for the context.
    :param device_id: a GPGPU device (and API) to use for the context.
    :param interactive: if ``True``, an interactive dialogue will be shown
        allowing one to choose the GPGPU device to use.
        If ``False``, the first device satisfying the filters (see below) will be chosen.
    :param api:
    :param include_devices:
    :param exclude_devices:
    :param include_platforms:
    :param exclude_platforms: see :py:func:`~nufhe.find_devices`.
    """

    def __init__(
            self, rng=None,
            thread=None,
            device_id: DeviceID=None,
            api=None, interactive=False,
            include_devices=None, exclude_devices=None,
            include_platforms=None, exclude_platforms=None):

        if rng is None:
            rng = DeterministicRNG()

        if thread is not None:
            # Use the given Thread object
            pass
        elif device_id is not None:
            api_obj, device = device_id.get_api_and_device()
            thread = api_obj.Thread(device)
        else:
            api_obj = _get_api_object(api)
            thread = api_obj.Thread.create(
                interactive=interactive,
                device_filters=dict(
                    include_devices=include_devices,
                    exclude_devices=exclude_devices,
                    include_platforms=include_platforms,
                    exclude_platforms=exclude_platforms))

        self.rng = rng
        self.thread = thread

    def __del__(self):
        # Comnputation cache retains some Thread-related objects,
        # so it needs to be cleared before Thread is destroyed.
        # This helps avoid CUDA errors in multi-thread usage.
        if hasattr(self, 'thread'):
            clear_computation_cache(self.thread)

    def make_secret_key(self, **params):
        """
        Creates a secret key, with ``params`` used to
        initialize a :py:class:`NuFHEParameters` object.

        The low-level analogue: :py:meth:`NuFHESecretKey.from_rng`.

        :returns: a :py:class:`NuFHESecretKey` object.
        """
        nufhe_params = NuFHEParameters(**params)
        return NuFHESecretKey.from_rng(self.thread, nufhe_params, self.rng)

    def make_cloud_key(self, secret_key: NuFHESecretKey):
        """
        Creates a cloud key matching the given secret key.

        The low-level analogue: :py:meth:`NuFHECloudKey.from_rng`.

        :returns: a :py:class:`NuFHECloudKey` object.
        """
        return NuFHECloudKey.from_rng(self.thread, secret_key.params, self.rng, secret_key)

    def make_key_pair(self, **params):
        """
        Creates a pair of a secret key and a matching cloud key.

        The low-level analogue: :py:func:`make_key_pair`.

        :returns: a tuple of a :py:class:`NuFHESecretKey` and a :py:class:`NuFHECloudKey` objects.
        """
        secret_key = self.make_secret_key(**params)
        cloud_key = self.make_cloud_key(secret_key)
        return secret_key, cloud_key

    def encrypt(self, secret_key: NuFHESecretKey, message):
        """
        Encrypts a message (a list or a ``numpy`` array treated as an array of booleans).

        The low-level analogue: :py:func:`encrypt`.

        :returns: an :py:class:`LweSampleArray` object with the same `shape` as the given array.
        """
        return encrypt(self.thread, self.rng, secret_key, message)

    def decrypt(self, secret_key: NuFHESecretKey, ciphertext: LweSampleArray):
        """
        Decrypts a message.

        The low-level analogue: :py:func:`decrypt`.

        :returns: a ``numpy.ndarray`` object of the type ``numpy.bool``
            and the same `shape` as ``ciphertext``.
        """
        return decrypt(self.thread, secret_key, ciphertext)

    def make_virtual_machine(
            self, cloud_key: NuFHECloudKey, perf_params: PerformanceParameters=None):
        """
        Creates an FHE "virtual machine" which can execute logical gates using the given cloud key.
        Optionally, one can pass a :py:class:`PerformanceParameters` object which will be
        specialized for the GPU device of the context and used in all the gate calls.

        :returns: a :py:class:`~nufhe.api_high_level.VirtualMachine` object.
        """
        return VirtualMachine(self.thread, cloud_key, perf_params=perf_params)

    def load_ciphertext(self, file_or_bytestring):
        """
        Load a ciphertext (a :py:class:`LweSampleArray` object) serialized with
        :py:meth:`LweSampleArray.dump` or :py:meth:`LweSampleArray.dumps`
        into the context memory space.

        The low-level analogues: :py:meth:`LweSampleArray.load` and :py:meth:`LweSampleArray.loads`.

        :returns: an :py:class:`LweSampleArray` object
        """
        if isinstance(file_or_bytestring, bytes):
            return LweSampleArray.loads(file_or_bytestring, self.thread)
        else:
            return LweSampleArray.load(file_or_bytestring, self.thread)

    def load_secret_key(self, file_or_bytestring):
        """
        Load a secret key (a :py:class:`NuFHESecretKey` object) serialized with
        :py:meth:`NuFHESecretKey.dump` or :py:meth:`NuFHESecretKey.dumps`
        into the context memory space.

        The low-level analogues: :py:meth:`NuFHESecretKey.load` and :py:meth:`NuFHESecretKey.loads`.

        :returns: a :py:class:`NuFHESecretKey` object
        """
        if isinstance(file_or_bytestring, bytes):
            return NuFHESecretKey.loads(file_or_bytestring, self.thread)
        else:
            return NuFHESecretKey.load(file_or_bytestring, self.thread)

    def load_cloud_key(self, file_or_bytestring):
        """
        Load a secret key (a :py:class:`NuFHECloudKey` object) serialized with
        :py:meth:`NuFHECloudKey.dump` or :py:meth:`NuFHECloudKey.dumps`
        into the context memory space.

        The low-level analogues: :py:meth:`NuFHECloudKey.load` and :py:meth:`NuFHECloudKey.loads`.

        :returns: a :py:class:`NuFHECloudKey` object
        """
        if isinstance(file_or_bytestring, bytes):
            return NuFHECloudKey.loads(file_or_bytestring, self.thread)
        else:
            return NuFHECloudKey.load(file_or_bytestring, self.thread)


class VirtualMachine:
    """
    A fully encrypted virtual machine capable of executing gates on ciphertexts
    (:py:class:`~nufhe.LweSampleArray` objects) using an encapsulated cloud key.

    .. method:: gate_<operator>(*args, dest: LweSampleArray=None)

        Calls one of :ref:`logical-gates`, using the context, the cloud key,
        and the performance parameters of the virtual machine.

        If ``dest`` is ``None``, creates a new ciphertext and uses it
        to store the output of the gate;
        otherwise ``dest`` is used for that purpose.

        :returns: an :py:class:`~nufhe.LweSampleArray` object
            with the result of the gate application.
    """

    def __init__(self, thread, cloud_key: NuFHECloudKey, perf_params: PerformanceParameters=None):
        "__init__()"
        if perf_params is None:
            perf_params = PerformanceParameters(cloud_key.params)

        perf_params = perf_params.for_device(thread.device_params)

        self.thread = thread
        self.params = cloud_key.params
        self.cloud_key = cloud_key
        self.perf_params = perf_params

    def empty_ciphertext(self, shape):
        """
        Returns an unitialized ciphertext (an :py:class:`~nufhe.LweSampleArray` object).

        The low-level analogue: :py:func:`empty_ciphertext`.
        """
        return empty_ciphertext(self.thread, self.params, shape)

    def load_ciphertext(self, file):
        """
        Load a ciphertext (a :py:class:`~nufhe.LweSampleArray` object) serialized with
        :py:meth:`LweSampleArray.dump <nufhe.LweSampleArray.dump>` into the context memory space.

        The low-level analogue: :py:meth:`LweSampleArray.load <nufhe.LweSampleArray.load>`.

        :returns: an :py:class:`~nufhe.LweSampleArray` object
        """
        return LweSampleArray.load(file, self.thread)

    def _gate(self, name, *args, dest: LweSampleArray=None):
        if dest is None:
            shapes = [get_shape(arg) for arg in args]
            dest = self.empty_ciphertext(result_shape(*shapes))
        gate_func = getattr(gates, name)
        gate_func(self.thread, self.cloud_key, dest, *args, perf_params=self.perf_params)
        return dest

    def __getattr__(self, name):
        if name.startswith('gate_'):
            return lambda *args, **kwds: self._gate(name, *args, **kwds)
        else:
            raise AttributeError(name)
