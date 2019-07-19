---------------
Version history
---------------


0.0.3 (19 Jul 2019)
~~~~~~~~~~~~~~~~~~~

* ADDED: ``LweSampleArray.copy()`` for cloning a ciphertext.

* ADDED: ``LweSampleArray.roll()`` that cyclically shifts encrypted bits if a ciphertext.

* ADDED: ``thread`` keyword parameter to ``Context``, allowing one to use an existing Reikna ``Thread`` object to create a context.

* ADDED: ``concatenate()`` for ``LweSampleArray`` objects.

* ADDED: ``__setitem__()`` functionality for ``LweSampleArray`` objects (the source must be another ``LweSampleArray``).

* ADDED: NTT transform now uses Montgomery multiplication for the cases where one of the factors can be prepared in advance, increasing performance (mostly for the multi-kernel bootstrap).

* FIXED: result shape derivation in gate methods of ``VirtualMachine``, including ``vm.gate_constant()`` not accepting lists as arguments.


0.0.2 (14 Feb 2019)
~~~~~~~~~~~~~~~~~~~

* **CHANGED:** a ``PerformanceParameters`` object needs to be specialized for the device used (by calling its ``for_device()`` method) before passing it to gates.

* **CHANGED:** instead of using ``numpy.random.RandomState`` for key generation and encryption, ``DeterministicRNG`` and ``SecureRNG`` are available instead. The former is the wrapped ``RandomState``, fast, but not cryptographically secure; the latter is the secure random source provided by the OS, which can be rather slow.

* ADDED: a high-level API hiding the Reikna details and removing some boilerplate.

* ADDED: shape checks in gate functions that take into account possible broadcasting.

* ADDED: ``dumps()`` and ``loads()`` methods for ``NuFHESecretKey``, ``NuFHECloudKey`` and ``LweSampleArray`` for serializing to/from bytestrings. The ``Context``'s ``load_secret_key``, ``load_cloud_key`` and ``load_ciphertext`` also take bytestrings as arguments.

* ADDED: exposed ``clear_computation_cache()`` which helps release the resources associated with a GPU context (the NuFHE ``Context`` objects call it automatically on destruction).

* ADDED: a ``find_devices()`` function to help with using multiple computation devices, and a corresponding keyword ``device_id`` for ``Context`` class constructor that uses its return values.

* ADDED: an example of multi-threaded multi-GPU usage.

* FIXED: a bug in ``tlwe_noiseless_trivial()`` occasionally leading to memory corruption.

* FIXED: a bug where ``PerformanceParameters`` and ``PerformanceParametersForDevice`` objects did not have a correct equality implementation, leading to unnecessary re-compilation of kernels.

* FIXED: compilation failing when ``transforms_per_block`` in ``PerformanceParameters`` is set too high.


0.0.1 (12 Oct 2018)
~~~~~~~~~~~~~~~~~~~

Initial version.
