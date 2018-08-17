import numpy
import nufhe
from reikna.cluda import any_api

thr = any_api().Thread.create(interactive=True)

rng = numpy.random.RandomState()
private_key, public_key = nufhe.nufhe_key_pair(thr, rng)

size = 32

bits1 = rng.randint(0, 2, size=size).astype(numpy.bool)
bits2 = rng.randint(0, 2, size=size).astype(numpy.bool)

ciphertext1 = nufhe.nufhe_encrypt(thr, rng, private_key, bits1)
ciphertext2 = nufhe.nufhe_encrypt(thr, rng, private_key, bits2)

reference = ~(bits1 * bits2)

result = nufhe.empty_ciphertext(thr, public_key.params, ciphertext1.shape)
nufhe.nufhe_gate_NAND_(thr, public_key, result, ciphertext1, ciphertext2)

result_bits = nufhe.nufhe_decrypt(thr, private_key, result)
assert (result_bits == reference).all()
