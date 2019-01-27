"""
This example demonstrates how to build a simple multi-threaded application,
where each thread uses a separate GPU to apply an FHE gate to a part of a ciphertext.

In order to run it you need two GPGPU devices available, both of them supporting
the transform type you are planning to use (because they will have to use the same cloud key).

This example uses multi-threading for simplicity, but multi-processing
(using either the Python module or MPI) is also possible.
Note that CUDA may have some restrictions, for example, you cannot have a CUDA
context both in the main process and the child processes when using Python multithreading.
"""

from threading import Thread
from queue import Queue

import random
import nufhe


class MyThread:
    """
    A simple wrapper that allows one to receive the value
    returned from the thread worker transparently.
    """

    def __init__(self, target, args=()):
        self.return_queue = Queue()
        self.target = target
        self.thread = Thread(target=self._target_wrapper, args=args)

    def _target_wrapper(self, *args):
        res = self.target(*args)
        self.return_queue.put(res)

    def start(self):
        self.thread.start()
        return self

    def join(self):
        ret_val = self.return_queue.get()
        self.thread.join()
        return ret_val


def worker(device_id, cloud_key_cpu, ciphertext1_cpu, ciphertext2_cpu):
    """
    The thread worker function.
    Runs a NAND gate over two provided ciphertexts and returns the serialized result.
    """
    print("Running a thread with", device_id)

    ctx = nufhe.Context(device_id=device_id)

    cloud_key = ctx.load_cloud_key(cloud_key_cpu)
    ciphertext1 = ctx.load_ciphertext(ciphertext1_cpu)
    ciphertext2 = ctx.load_ciphertext(ciphertext2_cpu)

    vm = ctx.make_virtual_machine(cloud_key)
    result = vm.gate_nand(ciphertext1, ciphertext2)
    result_cpu = result.dumps()

    print("Done")

    return result_cpu


if __name__ == '__main__':

    # This part is identical to the `gate_nand` example:
    # create the key pair and encrypt some data.

    size = 32
    bits1 = [random.choice([False, True]) for i in range(size)]
    bits2 = [random.choice([False, True]) for i in range(size)]
    reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]

    ctx = nufhe.Context()
    secret_key, cloud_key = ctx.make_key_pair()

    ciphertext1 = ctx.encrypt(secret_key, bits1)
    ciphertext2 = ctx.encrypt(secret_key, bits2)

    # Serialize the cloud key to pass it to child threads.

    ck = cloud_key.dumps()

    # Split ciphertexts into two parts each and serialize them.

    ct1_part1 = ciphertext1[:size//2].dumps()
    ct1_part2 = ciphertext1[size//2:].dumps()
    ct2_part1 = ciphertext2[:size//2].dumps()
    ct2_part2 = ciphertext2[size//2:].dumps()

    # Start two threads each applying NAND gate to their respective parts of the ciphertext.

    devices = nufhe.find_devices()

    assert len(devices) >= 2

    t1 = MyThread(target=worker, args=(devices[0], ck, ct1_part1, ct2_part1)).start()
    t2 = MyThread(target=worker, args=(devices[1], ck, ct1_part2, ct2_part2)).start()
    result_part1 = t1.join()
    result_part2 = t2.join()

    result_part1 = ctx.load_ciphertext(result_part1)
    result_part2 = ctx.load_ciphertext(result_part2)

    # Decrypt the results, join and test against the reference

    r1 = ctx.decrypt(secret_key, result_part1)
    r2 = ctx.decrypt(secret_key, result_part2)

    assert r1.tolist() + r2.tolist() == reference
