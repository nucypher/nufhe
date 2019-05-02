-------------
API reference
-------------

.. module:: nufhe


High-level api
~~~~~~~~~~~~~~

.. autoclass:: Context
    :members:

.. autofunction:: find_devices

.. autoclass:: nufhe.api_high_level.DeviceID

.. autoclass:: nufhe.api_high_level.VirtualMachine
    :members:


Parameters and keys
~~~~~~~~~~~~~~~~~~~

.. autoclass:: NuFHEParameters
    :members:

.. autoclass:: NuFHESecretKey
    :members:

.. autoclass:: NuFHECloudKey
    :members:

.. autofunction:: make_key_pair


.. _random-number-generators:

Random number generators
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeterministicRNG

.. autoclass:: SecureRNG


Encryption/decryption
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: encrypt

.. autofunction:: decrypt

.. autofunction:: empty_ciphertext

.. autoclass:: LweSampleArray
    :members:
    :special-members: __getitem__

.. autofunction:: concatenate

.. _logical-gates:

Logical gates
~~~~~~~~~~~~~

Unary gates
-----------

.. autofunction:: gate_constant

.. autofunction:: gate_copy

.. autofunction:: gate_not

Binary gates
------------

.. autofunction:: gate_and

.. autofunction:: gate_or

.. autofunction:: gate_xor

.. autofunction:: gate_nand

.. autofunction:: gate_nor

.. autofunction:: gate_xnor

.. autofunction:: gate_andny

.. autofunction:: gate_andyn

.. autofunction:: gate_orny

.. autofunction:: gate_oryn

Ternary gates
-------------

.. autofunction:: gate_mux


Performance parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PerformanceParameters
    :members: for_device

.. autoclass:: nufhe.performance.PerformanceParametersForDevice


Utility functions
~~~~~~~~~~~~~~~~~

.. autofunction:: clear_computation_cache
