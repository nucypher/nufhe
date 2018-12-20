import random
import numpy
import nufhe
from reikna.cluda import any_api

size = 32
bits1 = [random.choice([False, True]) for i in range(size)]
bits2 = [random.choice([False, True]) for i in range(size)]
reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]

thr = any_api().Thread.create(interactive=True)

rng = nufhe.DeterministicRNG()
secret_key, cloud_key = nufhe.make_key_pair(thr, rng)

ciphertext1 = nufhe.encrypt(thr, rng, secret_key, bits1)
ciphertext2 = nufhe.encrypt(thr, rng, secret_key, bits2)

result = nufhe.empty_ciphertext(thr, cloud_key.params, ciphertext1.shape)
nufhe.gate_nand(thr, cloud_key, result, ciphertext1, ciphertext2)

result_bits = nufhe.decrypt(thr, secret_key, result)

assert all(result_bits == reference)

