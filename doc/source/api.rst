-------------
API reference
-------------

.. module:: nufhe


Keys
~~~~

.. autoclass:: NuFHEParameters
    :members:

.. autoclass:: NuFHESecretKey
    :members:

.. autoclass:: NuFHECloudKey
    :members:

.. autofunction:: make_key_pair


Encryption/decryption
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: encrypt

.. autofunction:: decrypt

.. autofunction:: empty_ciphertext

.. autoclass:: LweSampleArray
    :members:
    :special-members: __getitem__


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
