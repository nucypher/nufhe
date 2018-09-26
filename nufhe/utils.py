import numpy


def arrays_equal(arr1, arr2):
    """
    Returns ``True`` if two integer arrays are equal.
    """

    # PyOpenCL arrays have an ``all()`` method, but ``PyCUDA`` ones don't,
    # and they're both hidden behind Reikna ``Array`` anyway.
    # Since this function is used for testing purposes only,
    # we just bring any GPU array to CPU before comparison.

    if not isinstance(arr1, numpy.ndarray):
        arr1 = arr1.get()

    if not isinstance(arr2, numpy.ndarray):
        arr2 = arr2.get()

    return (arr1 == arr2).all()
