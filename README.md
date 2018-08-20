# A GPU implementation of fully homomorphic encryption on torus

This library implements the fully homomorphic encryption algorithm from [`TFHE`](https://github.com/tfhe/tfhe) using CUDA and OpenCL. Unlike `TFHE`, where FFT is used internally to speed up polynomial multiplication, `nuFHE` can use either FFT or purely integer NTT (DFT-like transform on a finite field). The latter is based on the arithmetic operations and NTT scheme from [`cuFHE`](https://github.com/vernamlab/cuFHE).

<table>
  <tr>
    <td rowspan="2"><b>Platform</b></td>
    <td rowspan="2"><b>Library</b></td>
    <td colspan="2"><b>Performance (ms/bit)</b></td>
  </tr>
  <tr>
    <td><b>Binary Gate</b></td>
    <td><b>MUX Gate</b></td>
  </tr>
  <tr>
    <td rowspan="3"><b>Single Core/Single GPU - FFT</b></td>
    <td>TFHE (CPU)</td>
    <td>13</td>
    <td>26</td>
  </tr>
  <tr>
    <td>nuFHE</td>
    <td>0.13</td>
    <td>0.22</td>
  </tr>
  <tr>
    <td><b>Speedup</b></td>
    <td><b>100.9</b></td>
    <td><b>117.7</b></td>
  </tr>
  <tr>
    <td rowspan="3"><b>Single Core/Single GPU - NTT</b></td>
    <td>cuFHE</td>
    <td>0.35</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>nuFHE</td>
    <td>0.35</td>
    <td>0.67</td>
  </tr>
  <tr>
    <td><b>Speedup</b></td>
    <td><b>1.0</b></td>
    <td><b>-</b></td>
  </tr>
</table>

## Usage example

The code from this example can be found in `examples/gate_nand.py`.


### GPU thread

`nufhe` uses [`reikna`](https://github.com/fjarri/reikna) as a backend for GPU operations, and all the `nufhe` calls require a `reikna` [`Thread`](http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread) object, encapsulating a GPU context and a serialization queue for GPU kernel calls. It can be created interactively:

    from reikna.cluda import cuda_api

    thr = cuda_api().Thread.create(interactive=True)

where the user will be offered to choose between available platforms and videocards. Alternatively, one can pick an arbitrary available platform/device:

    thr = cuda_api().Thread.create()

It is also possible to create a `Thread` using a known device, or existing [`PyCUDA`](https://github.com/inducer/pycuda) or [`PyOpenCL`](https://github.com/inducer/pyopencl) context. This is advanced usage, for those who plan to integrate `nuFHE` into a larger GPU-based program. See the documentation for [`Thread`](http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread) and [`Thread.create()`](http://reikna.publicfields.net/en/latest/api/cluda.html#reikna.cluda.api.Thread.create) for details.

If one wants to use OpenCL instead of CUDA, `cuda_api` should be replaced with `ocl_api`. Alternatively, one can use `any_api` to select an arbitrary available one.


### Key pair

The next step is the creation of a private and a public key. The former is used to encrypt plaintexts or decrypt cyphertexts; the latter is required to apply gates to encrypted data. Note that the public key can be rather large (on the order of 100Mb).

    import numpy
    import nufhe

    rng = numpy.random.RandomState()
    private_key, public_key = nufhe.make_key_pair(thr, rng)

`make_key_pair` takes some keyword parameters that affect the security of the algorithm; the default values correspond to about 110 bits of security.


### Encryption

Using the private key we can encrypt some data. `nuFHE` gates operate on bit arrays:

    size = 32

    bits1 = rng.randint(0, 2, size=size).astype(numpy.bool)
    bits2 = rng.randint(0, 2, size=size).astype(numpy.bool)

    ciphertext1 = nufhe.encrypt(thr, rng, private_key, bits1)
    ciphertext2 = nufhe.encrypt(thr, rng, private_key, bits2)

In this example we will test the NAND gate, so the reference result would be

    reference = ~(bits1 * bits2)


### Processing

On the server side, where only the public key is known, one can use it to apply a gate:

    result = nufhe.empty_ciphertext(thr, public_key.params, ciphertext1.shape)
    nufhe.gate_nand(thr, public_key, result, ciphertext1, ciphertext2)


### Decryption

After the processing, the person in possession of the private key can decrypt the result and verify that the gate was applied correctly:

    result_bits = nufhe.decrypt(thr, private_key, result)
    assert (result_bits == reference).all()
