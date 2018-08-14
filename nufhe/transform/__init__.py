modulus = 2**64 - 2**32 + 1

from .arithmetic import add, sub, mod, mul, pow, inv_pow2, lsh
from .ntt import ntt1024
from .fft import fft512
from .computation import Transform
