# Copyright (C) 2018 NuCypher
#
# This file is part of nufhe.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy

from reikna.helpers import product

from .numeric_functions import Torus32
from .polynomial_transform import get_transform


def TLweNoiselessTrivialReference(params: 'TLweParams', shape):

    mask_size = params.mask_size
    polynomial_degree = params.polynomial_degree
    batch_len = product(shape)

    def _kernel(a, current_variances, mu):
        a_view = a.reshape(batch_len, mask_size + 1, polynomial_degree)
        a_view[:,:mask_size,:] = 0
        a_view[:,mask_size,:] = mu.reshape(batch_len, polynomial_degree)
        current_variances.fill(0)

    return _kernel


def TLweExtractLweSamplesReference(params: 'TLweParams', shape):

    mask_size = params.mask_size
    polynomial_degree = params.polynomial_degree
    batch_len = product(shape)

    def _kernel(result_a, result_b, tlwe_a):

        batch = product(tlwe_a.shape[:-2])

        a_view = result_a.reshape(batch_len, mask_size, polynomial_degree)
        b_view = result_b.reshape(batch_len)
        tlwe_a_view = tlwe_a.reshape(batch_len, mask_size + 1, polynomial_degree)

        a_view[:,:,0] = tlwe_a_view[:, :mask_size, 0]
        a_view[:,:,1:] = -tlwe_a_view[:, :mask_size, :0:-1]

        numpy.copyto(b_view, tlwe_a_view[:, mask_size, 0])

    return _kernel


# create an homogeneous tlwe sample
def TLweEncryptZeroReference(params: 'TLweParams', shape, noise: float, perf_params):
    polynomial_degree = params.polynomial_degree
    mask_size = params.mask_size

    transform = get_transform(params.transform_type)
    tr_dtype = transform.transformed_dtype()

    def _kernel(result_a, result_cv, key, noises1, noises2):
        batch_len = product(shape)
        result_a_view = result_a.reshape(batch_len, mask_size + 1, polynomial_degree)
        noises1_view = noises1.reshape(batch_len, mask_size, polynomial_degree)
        noises2_view = noises2.reshape(batch_len, polynomial_degree)

        tmp1 = transform.forward_transform_ref(key)
        tmp2 = transform.forward_transform_ref(noises1_view)
        tmp3 = transform.transformed_space_mul_ref(tmp1, tmp2)
        tmpr = transform.inverse_transform_ref(tmp3)

        result_a_view[:,:mask_size,:] = noises1_view
        result_a_view[:,mask_size,:] = noises2_view
        for i in range(mask_size):
            result_a_view[:,mask_size,:] += tmpr[:,i,:]

        result_cv.fill(noise**2)

    return _kernel
