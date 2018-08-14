import numpy
import time

from nufhe import *

from reikna.cluda import cuda_api
api = cuda_api()
thr = api.Thread.create()

size = 32

def encrypt():

    rng = numpy.random.RandomState(123)

    print("Key generation:")
    t = time.time()
    secret_key, cloud_key = nufhe_key_pair(thr, rng)
    print(time.time() - t)

    bits1 = rng.randint(0, 2, size=size).astype(numpy.bool)
    bits2 = rng.randint(0, 2, size=size).astype(numpy.bool)

    print("Encryption:")
    t = time.time()
    ciphertext1 = nufhe_encrypt(thr, rng, secret_key, bits1)
    ciphertext2 = nufhe_encrypt(thr, rng, secret_key, bits2)
    print(time.time() - t)

    reference = numpy.logical_not(numpy.logical_and(bits1, bits2))

    return secret_key, cloud_key, ciphertext1, ciphertext2, reference


def process(cloud_key, ciphertext1, ciphertext2):
    params = nufhe_parameters(cloud_key)
    result = empty_ciphertext(thr, params, ciphertext1.shape_info.shape)

    nufhe_gate_NAND_(thr, cloud_key, result, ciphertext1, ciphertext2)
    thr.synchronize()

    #print(thr.temp_alloc._statistics())

    print("Processing:")
    t = time.time()
    nufhe_gate_NAND_(thr, cloud_key, result, ciphertext1, ciphertext2)
    thr.synchronize()
    t = time.time() - t
    print(t)
    print(t / size * 1000, "ms per bit")

    #import cProfile
    #cProfile.runctx("nufhe_gate_NAND_(cloud_key, result, ciphertext1, ciphertext2)",
    #    locals=locals(), globals=globals(), sort='cumtime')

    return result


def verify(secret_key, answer, reference):
    print("Decryption:")
    t = time.time()
    answer_bits = nufhe_decrypt(thr, secret_key, answer)
    print(time.time() - t)

    if (answer_bits == reference).all():
        print("Correct")
    else:
        print("Incorrect bits:", (answer_bits != reference).sum())
        print((answer_bits != reference).astype(numpy.int32))


secret_key, cloud_key, ciphertext1, ciphertext2, reference = encrypt()
answer = process(cloud_key, ciphertext1, ciphertext2)
verify(secret_key, answer, reference)
