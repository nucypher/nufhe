import os
import random
import nufhe


def client_prepare():

    size = 32
    bits1 = [random.choice([False, True]) for i in range(size)]
    bits2 = [random.choice([False, True]) for i in range(size)]
    reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]

    ctx = nufhe.Context()
    secret_key, cloud_key = ctx.make_key_pair()
    ciphertext1 = ctx.encrypt(secret_key, bits1)
    ciphertext2 = ctx.encrypt(secret_key, bits2)

    with open('secret_key', 'wb') as f:
        secret_key.dump(f)

    with open('cloud_key', 'wb') as f:
        cloud_key.dump(f)

    with open('ciphertext1', 'wb') as f:
        ciphertext1.dump(f)

    with open('ciphertext2', 'wb') as f:
        ciphertext2.dump(f)

    return reference


def cloud_process():

    ctx = nufhe.Context()

    with open('cloud_key', 'rb') as f:
        cloud_key = ctx.load_cloud_key(f)

    vm = ctx.make_virtual_machine(cloud_key)

    with open('ciphertext1', 'rb') as f:
        ciphertext1 = vm.load_ciphertext(f)

    with open('ciphertext2', 'rb') as f:
        ciphertext2 = vm.load_ciphertext(f)

    result = vm.gate_nand(ciphertext1, ciphertext2)

    with open('result', 'wb') as f:
        result.dump(f)


def client_verify(reference):

    ctx = nufhe.Context()

    with open('secret_key', 'rb') as f:
        secret_key = ctx.load_secret_key(f)

    with open('result', 'rb') as f:
        result = ctx.load_ciphertext(f)

    result_bits = ctx.decrypt(secret_key, result)
    assert all(result_bits == reference)


def cleanup():
    os.remove('secret_key')
    os.remove('cloud_key')
    os.remove('ciphertext1')
    os.remove('ciphertext2')
    os.remove('result')


reference = client_prepare()
cloud_process()
client_verify(reference)
cleanup()
