import numpy
import time

from tfhe import *

from reikna.cluda import cuda_api, ocl_api
api = cuda_api()
thr = api.Thread.create()


def int_to_bitarray(x):
    return numpy.array([((x >> i) & 1 != 0) for i in range(x.itemsize * 8)])

def ints_to_bitarray(xs):
    return numpy.vstack(int_to_bitarray(x) for x in xs)

def bitarray_to_ints(xs, dtype):
    ints = []
    for j in range(xs.shape[0]):
        int_answer = 0
        for i in range(16):
            int_answer = int_answer | (xs[j,i] << i)
        ints.append(int_answer)
    return numpy.array(ints, dtype)


def reference_mux(bits1, bits2, bits3):
    numpy.array([b if a else c for a, b, c in zip(bits1, bits2, bits3)])


def encrypt():

    rng = numpy.random.RandomState(123)
    secret_key, cloud_key = tfhe_key_pair(thr, rng)

    ints1 = rng.randint(0, 1000, size=4).astype(numpy.uint16)
    ints2 = rng.randint(0, 1000, size=4).astype(numpy.uint16)

    print(ints1)
    print(ints2)

    reference = numpy.array([min(x, y) for x, y in zip(ints1, ints2)], numpy.uint16)

    bits1 = ints_to_bitarray(ints1)
    bits2 = ints_to_bitarray(ints2)

    ciphertext1 = tfhe_encrypt(thr, rng, secret_key, bits1)
    ciphertext2 = tfhe_encrypt(thr, rng, secret_key, bits2)

    return secret_key, cloud_key, ciphertext1, ciphertext2, reference


# elementary full comparator gate that is used to compare the i-th bit:
#   input: ai and bi the i-th bit of a and b
#          lsb_carry: the result of the comparison on the lowest bits
#   algo: if (a==b) return lsb_carry else return b
def encrypted_compare_bit_(cloud_key, result, a, b, lsb_carry, tmp):
    tfhe_gate_XNOR_(thr, cloud_key, tmp, a, b)
    tfhe_gate_MUX_(thr, cloud_key, result, tmp, lsb_carry, a)


# this function compares two multibit words, and puts the max in result
def encrypted_minimum_(cloud_key, result, a, b):

    nb_bits = result.shape_info.shape[1]

    params = tfhe_parameters(cloud_key)

    tmp1 = empty_ciphertext(thr, params, a.shape_info.shape[:1] + (1,))
    tmp2 = empty_ciphertext(thr, params, a.shape_info.shape[:1] + (1,))

    # initialize the carry to 0
    tfhe_gate_CONSTANT_(thr, cloud_key, tmp1, False)

    # run the elementary comparator gate n times
    for i in range(nb_bits):
        encrypted_compare_bit_(cloud_key, tmp1, a[:,i:i+1], b[:,i:i+1], tmp1, tmp2)

    # tmp1 is the result of the comparaison: 0 if a is larger, 1 if b is larger
    # select the max and copy it to the result
    tfhe_gate_MUX_(thr, cloud_key, result, tmp1, b, a)



def process(cloud_key, ciphertext1, ciphertext2):

    # if necessary, the params are inside the key
    params = tfhe_parameters(cloud_key)

    # do some operations on the ciphertexts: here, we will compute the
    # minimum of the two
    result = empty_ciphertext(thr, params, ciphertext1.shape_info.shape)
    encrypted_minimum_(cloud_key, result, ciphertext1, ciphertext2)

    return result


def verify(secret_key, answer, reference):
    answer_bits = tfhe_decrypt(thr, secret_key, answer)
    ints_answer = bitarray_to_ints(answer_bits, reference.dtype)
    print("Answer:", ints_answer)
    print("Reference:", reference)


secret_key, cloud_key, ciphertext1, ciphertext2, reference = encrypt()
answer = process(cloud_key, ciphertext1, ciphertext2)
verify(secret_key, answer, reference)
