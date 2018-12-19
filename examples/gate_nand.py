import random
import nufhe

size = 32
bits1 = [random.choice([False, True]) for i in range(size)]
bits2 = [random.choice([False, True]) for i in range(size)]
reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]

ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()

ciphertext1 = ctx.encrypt(secret_key, bits1)
ciphertext2 = ctx.encrypt(secret_key, bits2)

vm = ctx.make_virtual_machine(cloud_key)
result = vm.gate_nand(ciphertext1, ciphertext2)
result_bits = ctx.decrypt(secret_key, result)

assert all(result_bits == reference)
