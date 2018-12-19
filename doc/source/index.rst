NuFHE, a GPU-powered Torus FHE implementation
=============================================

--------
Contents
--------

.. toctree::
   :maxdepth: 1

   implementation_details
   api
   history


------------
Introduction
------------

``nufhe`` implements the fully homomorphic encryption algorithm from `TFHE <https://github.com/tfhe/tfhe>`_ using CUDA and OpenCL. For the theoretical background one may refer to the works TFHE is based on:

* C. Gentry, A. Sahai, and B. Waters, `"Homomorphic encryption from learning with errors: Conceptually-simpler, asymptotically-faster, attribute-based." <https://link.springer.com/chapter/10.1007/978-3-642-40041-4_5>`_, Crypto 75-92 (2013);
* L. Ducas and D. Micciancio, `"FHEW: Bootstrapping homomorphic encryption in less than a second." <https://link.springer.com/chapter/10.1007/978-3-662-46800-5_24>`_, Eurocrypt 617-640 (2015);
* I. Chillotti, N. Gama, M. Georgieva, and M. Izabachène. `"Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds" <https://link.springer.com/chapter/10.1007/978-3-662-53887-6_1>`_, Asiacrypt 3--33 (2016).

For more details check out `this collection of papers on lattice cryptography <https://cseweb.ucsd.edu/~daniele/LatticeLinks/FHE.html>`_.

Some additional performance improvements employed by ``nufhe`` are described in `Implementation details <implementation-details>`_.


------------
Installation
------------

``nufhe`` supports two GPU backends, CUDA (via `PyCUDA <https://documen.tician.de/pycuda/>`_) and OpenCL (via `PyOpenCL <https://documen.tician.de/pyopencl/>`_). Neither of the backend packages can be installed by default, because, depending on the videocard, one of the platforms may be unavailable. Therefore, the user must pick one or more backends they want to use and request them explicitly during installation. A simple rule of thumb is to pick CUDA if you have an nVidia videocard, and OpenCL otherwise (although OpenCL will work with nVidia cards as well). Then ``nufhe`` can be installed using PyPi specifying the required extras.


For the CUDA backend:

::

    $ pip install nufhe[pycuda]

For the OpenCL backend:

::

    $ pip install nufhe[pyopencl]

For both CUDA and OpenCL backends:

::

    $ pip install nufhe[pycuda,pyopencl]


---------------
A short example
---------------

.. module:: nufhe

::

    import random
    import nufhe

    ctx = nufhe.Context()
    secret_key, cloud_key = ctx.make_key_pair()

    size = 32
    bits1 = [random.choice([False, True]) for i in range(size)]
    bits2 = [random.choice([False, True]) for i in range(size)]

    ciphertext1 = ctx.encrypt(secret_key, bits1)
    ciphertext2 = ctx.encrypt(secret_key, bits2)

    reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]

    vm = ctx.make_virtual_machine(cloud_key)
    result = vm.gate_nand(ciphertext1, ciphertext2)
    result_bits = ctx.decrypt(secret_key, result)

    assert all(result_bits == reference)


Context
~~~~~~~

::

    ctx = nufhe.Context()

A context object represents an execution environment on a GPU (akin to a process), and is tied to a specific GPU device (if there are several available).
The target device can be either selected interactively, or picked automaticall based on various filters; see the :py:class:`Context` constructor for details.

Similar to a process, each context has its own memory space, and objects (keys and ciphertexts) from one context cannot be used in another one directly.
One can transfer them between contexts via serialization/deserialization, see :py:meth:`NuFHESecretKey.dump`, :py:meth:`NuFHECloudKey.dump` and :py:meth:`LweSampleArray.dump` for details.


Key pair
~~~~~~~~

The next step is the creation of a secret and a cloud key. The former is used to encrypt plaintexts or decrypt cyphertexts; the latter is required to apply gates to encrypted data. Note that the cloud key can be rather large (of the order of 100Mb).

::

    secret_key, cloud_key = ctx.make_key_pair()

:py:meth:`~Context.make_key_pair()` takes some keyword parameters that affect the security of the algorithm; the default values correspond to about 110 bits of security.


Encryption
~~~~~~~~~~

Using the secret key we can encrypt some data with :py:meth:`~Context.encrypt()`. ``nufhe`` gates operate on bit arrays (either lists or ``numpy`` arrays):

::

    size = 32
    bits1 = [random.choice([False, True]) for i in range(size)]
    bits2 = [random.choice([False, True]) for i in range(size)]

    ciphertext1 = ctx.encrypt(secret_key, bits1)
    ciphertext2 = ctx.encrypt(secret_key, bits2)

In this example we will test the NAND gate, so the reference result would be

::

    reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]


Processing
~~~~~~~~~~

Calculations are performed on a fully encrypted virtual machine created out of a cloud key:

::

    vm = ctx.make_virtual_machine(cloud_key)
    result = vm.gate_nand(ciphertext1, ciphertext2)

The output of a gate can be pre-initialized with :py:meth:`~nufhe.api_high_level.VirtualMachine.empty_ciphertext()` and passed to any gate function as a ``dest`` keyword parameter.


Decryption
~~~~~~~~~~

After the processing, the person in possession of the secret key can decrypt the result with :py:meth:`~Context.decrypt()` and verify that the gate was applied correctly:

::

    result_bits = ctx.decrypt(secret_key, result)
    assert all(result_bits == reference)


GPU threads for the low-level API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nufhe`` uses `Reikna <https://github.com/fjarri/reikna>`_ as a backend for GPU operations, and all the low-level ``nufhe`` calls require a ``reikna`` `Thread <http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread>`_ object, encapsulating a GPU context and a serialization queue for GPU kernel calls. It can be created interactively:

::

    from reikna.cluda import cuda_api

    thr = cuda_api().Thread.create(interactive=True)

where the user will be offered to choose between available platforms and videocards. Alternatively, one can pick an arbitrary available platform/device:

::

    thr = cuda_api().Thread.create()

It is also possible to create a ``Thread`` using a known device, or an existing `PyCUDA <https://github.com/inducer/pycuda>`_ or `PyOpenCL <https://github.com/inducer/pyopencl>`_ context. This is advanced usage, for those who plan to integrate ``nufhe`` into a larger GPU-based program. See the documentation for `Thread <http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread>`_ and `Thread.create() <http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread.create>`_ for details.

If one wants to use OpenCL instead of CUDA, ``cuda_api`` should be replaced with ``ocl_api``. Alternatively, one can use ``any_api`` to select an arbitrary available backend.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
