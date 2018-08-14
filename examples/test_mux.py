import numpy
import time

import nufhe
from nufhe import *

from reikna.cluda import cuda_api
api = cuda_api()
thr = api.Thread.create(async=True)


def int_to_bitarray(x, size=16):
    return numpy.array([((x >> i) & 1 != 0) for i in range(size)])


def bitarray_to_int(x):
    int_answer = 0
    for i in range(x.size):
        int_answer = int_answer | (x[i] << i)
    return int_answer


def reference_mux(bits1, bits2, bits3):
    numpy.array([b if a else c for a, b, c in zip(bits1, bits2, bits3)])


def encrypt():

    rng = numpy.random.RandomState(123)

    print("Key generation:")
    t = time.time()
    secret_key, cloud_key = nufhe_key_pair(rng)
    print(time.time() - t)

    size = 32

    bits1 = int_to_bitarray(2017, size=size)
    bits2 = int_to_bitarray(42, size=size)
    bits3 = int_to_bitarray(12345, size=size)

    print("Encryption:")
    t = time.time()
    ciphertext1 = nufhe_encrypt(rng, secret_key, bits1)
    ciphertext2 = nufhe_encrypt(rng, secret_key, bits2)
    ciphertext3 = nufhe_encrypt(rng, secret_key, bits3)
    print(time.time() - t)

    return secret_key, cloud_key, ciphertext1, ciphertext2, ciphertext3


def process(cloud_key, ciphertext1, ciphertext2, ciphertext3):
    params = nufhe_parameters(cloud_key)
    result = empty_ciphertext(params, ciphertext1.shape)

    cloud_key.to_gpu(thr)
    ciphertext1.to_gpu(thr)
    ciphertext2.to_gpu(thr)
    ciphertext3.to_gpu(thr)
    result.to_gpu(thr)

    #import cProfile
    #cProfile.runctx("nufhe_gate_MUX_(cloud_key, result, ciphertext1, ciphertext2, ciphertext3)",
    #    locals=locals(), globals=globals(), sort='cumtime')

    nufhe_gate_MUX_(cloud_key, result, ciphertext1, ciphertext2, ciphertext3)
    thr.synchronize()

    print("Processing:")
    t = time.time()
    nufhe_gate_MUX_(cloud_key, result, ciphertext1, ciphertext2, ciphertext3)
    thr.synchronize()
    print(time.time() - t)

    cloud_key.from_gpu()
    ciphertext1.from_gpu()
    ciphertext2.from_gpu()
    ciphertext3.from_gpu()
    result.from_gpu()

    return result


def verify(secret_key, answer):
    print("Decryption:")
    t = time.time()
    answer_bits = nufhe_decrypt(secret_key, answer)
    print(time.time() - t)

    int_answer = bitarray_to_int(answer_bits)
    print("Answer:", int_answer)


secret_key, cloud_key, ciphertext1, ciphertext2, ciphertext3 = encrypt()
answer = process(cloud_key, ciphertext1, ciphertext2, ciphertext3)
verify(secret_key, answer)
