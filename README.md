# A GPU implementation of fully homomorphic encryption on torus

This library implements the fully homomorphic encryption algorithm from [`TFHE`](https://github.com/tfhe/tfhe) using CUDA and OpenCL. Unlike `TFHE`, where FFT is used internally to speed up polynomial multiplication, `nufhe` can use either FFT or purely integer NTT (DFT-like transform on a finite field). The latter is based on the arithmetic operations and NTT scheme from [`cuFHE`](https://github.com/vernamlab/cuFHE). Refer to the [project documentation](https://nufhe.readthedocs.io/en/latest/) for more details.


## Usage example

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


## Performance

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
