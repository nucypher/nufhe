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

::

    import numpy
    import nufhe
    from reikna.cluda import any_api

    thr = any_api().Thread.create(interactive=True)

    rng = numpy.random.RandomState()
    secret_key, cloud_key = nufhe.make_key_pair(thr, rng)

    size = 32

    bits1 = rng.randint(0, 2, size=size).astype(numpy.bool)
    bits2 = rng.randint(0, 2, size=size).astype(numpy.bool)

    ciphertext1 = nufhe.encrypt(thr, rng, secret_key, bits1)
    ciphertext2 = nufhe.encrypt(thr, rng, secret_key, bits2)

    reference = ~(bits1 * bits2)

    result = nufhe.empty_ciphertext(thr, cloud_key.params, ciphertext1.shape)
    nufhe.gate_nand(thr, cloud_key, result, ciphertext1, ciphertext2)

    result_bits = nufhe.decrypt(thr, secret_key, result)
    assert (result_bits == reference).all()


GPU thread
~~~~~~~~~~

``nufhe`` uses `Reikna <https://github.com/fjarri/reikna>`_ as a backend for GPU operations, and all the ``nufhe`` calls require a ``reikna`` `Thread <http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread>`_ object, encapsulating a GPU context and a serialization queue for GPU kernel calls. It can be created interactively:

::

    from reikna.cluda import cuda_api

    thr = cuda_api().Thread.create(interactive=True)

where the user will be offered to choose between available platforms and videocards. Alternatively, one can pick an arbitrary available platform/device:

::

    thr = cuda_api().Thread.create()

It is also possible to create a ``Thread`` using a known device, or an existing `PyCUDA <https://github.com/inducer/pycuda>`_ or `PyOpenCL <https://github.com/inducer/pyopencl>`_ context. This is advanced usage, for those who plan to integrate ``nufhe`` into a larger GPU-based program. See the documentation for `Thread <http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread>`_ and `Thread.create() <http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread.create>`_ for details.

If one wants to use OpenCL instead of CUDA, ``cuda_api`` should be replaced with ``ocl_api``. Alternatively, one can use `any_api` to select an arbitrary available backend.


Key pair
~~~~~~~~

The next step is the creation of a secret and a cloud key. The former is used to encrypt plaintexts or decrypt cyphertexts; the latter is required to apply gates to encrypted data. Note that the cloud key can be rather large (on the order of 100Mb).

::

    import numpy
    import nufhe

    rng = numpy.random.RandomState()
    secret_key, cloud_key = nufhe.make_key_pair(thr, rng)

:py:func:`~nufhe.make_key_pair()` takes some keyword parameters that affect the security of the algorithm; the default values correspond to about 110 bits of security.


Encryption
~~~~~~~~~~

Using the secret key we can encrypt some data with :py:func:`~nufhe.encrypt()`. ``nufhe`` gates operate on bit arrays:

::

    size = 32

    bits1 = rng.randint(0, 2, size=size).astype(numpy.bool)
    bits2 = rng.randint(0, 2, size=size).astype(numpy.bool)

    ciphertext1 = nufhe.encrypt(thr, rng, secret_key, bits1)
    ciphertext2 = nufhe.encrypt(thr, rng, secret_key, bits2)

In this example we will test the NAND gate, so the reference result would be

::

    reference = ~(bits1 * bits2)


Processing
~~~~~~~~~~

On the server side, where only the cloud key is known, one can use it to apply a gate:

::

    result = nufhe.empty_ciphertext(thr, cloud_key.params, ciphertext1.shape)
    nufhe.gate_nand(thr, cloud_key, result, ciphertext1, ciphertext2)

Note that we had to initialize an empty output ciphertext with :py:func:`~nufhe.empty_ciphertext()`.


Decryption
~~~~~~~~~~

After the processing, the person in possession of the secret key can decrypt the result with :py:func:`~nufhe.decrypt()` and verify that the gate was applied correctly:

::

    result_bits = nufhe.decrypt(thr, secret_key, result)
    assert (result_bits == reference).all()


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
