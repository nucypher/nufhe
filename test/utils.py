import numpy
import pytest

import tfhe.transform.ntt_cpu as ntt_cpu


def tp_dtype(tp):
    if isinstance(tp, str) and tp == 'ff_number':
        return numpy.uint64
    else:
        return tp


def tp_limits(tp):
    if isinstance(tp, str) and tp == 'ff_number':
        return 0, ntt_cpu.GaloisNumber.modulus
    elif numpy.issubdtype(tp, numpy.integer):
        ii = numpy.iinfo(tp)
        return ii.min, ii.max + 1
    else:
        return -10, 10


def get_test_array(shape, tp, val_range=None):
    dtype = tp_dtype(tp)
    if val_range is None:
        nmin, nmax = tp_limits(tp)
    else:
        nmin, nmax = val_range

    if numpy.issubdtype(dtype, numpy.integer):
        return numpy.random.randint(nmin, nmax, dtype=dtype, size=shape)
    elif numpy.issubdtype(dtype, numpy.floating):
        return numpy.random.uniform(nmin, nmax, size=shape).astype(dtype)
    elif numpy.issubdtype(dtype, numpy.complexfloating):
        return (
            numpy.random.uniform(nmin, nmax, size=shape)
            + 1j * numpy.random.uniform(nmin, nmax, size=shape)).astype(dtype)
    else:
        raise NotImplementedError(dtype)
