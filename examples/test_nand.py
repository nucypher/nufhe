import numpy
import time

import tfhe
from tfhe import *

from reikna.cluda import cuda_api
from reikna.cluda.tempalloc import ZeroOffsetManager
api = cuda_api()
thr = api.Thread.create(async=True, temp_alloc=dict(pack_on_alloc=False))





size = 512


def encrypt():

    rng = numpy.random.RandomState(123)

    print("Key generation:")
    t = time.time()
    secret_key, cloud_key = tfhe_key_pair(rng)
    print(time.time() - t)

    bits1 = numpy.random.randint(0, 2, size=size).astype(numpy.bool)
    bits2 = numpy.random.randint(0, 2, size=size).astype(numpy.bool)

    print("Encryption:")
    t = time.time()
    ciphertext1 = tfhe_encrypt(rng, secret_key, bits1)
    ciphertext2 = tfhe_encrypt(rng, secret_key, bits2)
    print(time.time() - t)

    reference = numpy.logical_not(numpy.logical_and(bits1, bits2))

    return secret_key, cloud_key, ciphertext1, ciphertext2, reference


def process(cloud_key, ciphertext1, ciphertext2):
    params = tfhe_parameters(cloud_key)
    result = empty_ciphertext(params, ciphertext1.shape)

    cloud_key.to_gpu(thr)
    ciphertext1.to_gpu(thr)
    ciphertext2.to_gpu(thr)
    result.to_gpu(thr)

    tfhe_gate_NAND_(cloud_key, result, ciphertext1, ciphertext2)
    thr.synchronize()
    tfhe.lwe_bootstrapping.to_gpu_time = 0

    print(thr.temp_alloc._statistics())

    print("Processing:")
    t = time.time()
    tfhe_gate_NAND_(cloud_key, result, ciphertext1, ciphertext2)
    thr.synchronize()
    t = time.time() - t
    print(t)
    print(t / size * 1000, "ms per bit")
    print("minus to_gpu time, ", (t - tfhe.lwe_bootstrapping.to_gpu_time) / size * 1000, "ms per bit")

    #import cProfile
    #cProfile.runctx("tfhe_gate_NAND_(cloud_key, result, ciphertext1, ciphertext2)",
    #    locals=locals(), globals=globals(), sort='cumtime')

    cloud_key.from_gpu()
    ciphertext1.from_gpu()
    ciphertext2.from_gpu()
    result.from_gpu()

    return result


def verify(secret_key, answer, reference):
    print("Decryption:")
    t = time.time()
    answer_bits = tfhe_decrypt(secret_key, answer)
    print(time.time() - t)

    if (answer_bits == reference).all():
        print("Correct")
    else:
        print("Incorrect bits:", (answer_bits != reference).sum())


secret_key, cloud_key, ciphertext1, ciphertext2, reference = encrypt()
answer = process(cloud_key, ciphertext1, ciphertext2)
verify(secret_key, answer, reference)
