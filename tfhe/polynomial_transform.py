from . import polynomial_transform_fft
from . import polynomial_transform_ntt


def get_transform(transform_type):
    if transform_type == 'FFT':
        return polynomial_transform_fft
    elif transform_type == 'NTT':
        return polynomial_transform_ntt
