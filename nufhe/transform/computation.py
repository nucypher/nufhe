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

import reikna.helpers as helpers
from reikna.cluda import OutOfResourcesError
from reikna.core import Computation, Parameter, Annotation, Type


TEMPLATE = helpers.template_for(__file__)


class Transform(Computation):

    def __init__(
            self, transform, batch_shape, inverse=False,
            i32_conversion=False, transforms_per_block=4, kernel_repetitions=1):

        self._inverse = inverse
        self._transform = transform
        self._transforms_per_block = transforms_per_block
        self._kernel_repetitions = kernel_repetitions
        self._i32_conversion = i32_conversion

        tr_arr = Type(self._transform.elem_dtype, batch_shape + (transform.transform_length,))
        if i32_conversion:
            arr = Type(numpy.int32, batch_shape + (transform.polynomial_length,))
            if inverse:
                oarr = arr
                iarr = tr_arr
            else:
                oarr = tr_arr
                iarr = arr
        else:
            oarr = tr_arr
            iarr = tr_arr

        Computation.__init__(self, [
            Parameter('output', Annotation(oarr, 'o')),
            Parameter('input', Annotation(iarr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        batch_size = helpers.product(output.shape[:-1])

        cdata_arr = self._transform.cdata_inv if self._inverse else self._transform.cdata_fw
        if self._transform.use_constant_memory:
            cdata = plan.constant_array(cdata_arr)
        else:
            cdata = plan.persistent_array(cdata_arr)

        tpb = self._transforms_per_block
        while tpb >= 1:
            blocks_num = helpers.min_blocks(batch_size, tpb)
            try:
                plan.kernel_call(
                    TEMPLATE.get_def('standalone_transform'),
                        [output, input_, cdata],
                        kernel_name="standalone_transform",
                        global_size=(
                            blocks_num,
                            self._transform.threads_per_transform * tpb),
                        local_size=(
                            1,
                            self._transform.threads_per_transform * tpb),
                        render_kwds=dict(
                            inverse=self._inverse,
                            i32_conversion=self._i32_conversion,
                            kernel_repetitions=self._kernel_repetitions,
                            transform=self._transform,
                            transforms_per_block=tpb,
                            batch_size=batch_size,
                            blocks_num=blocks_num,
                            slices=(len(output.shape) - 1, 1)))
                break
            except OutOfResourcesError:
                tpb -= 1
        else:
            raise Exception(
                "The selected device does not have enough resources for the selected transform")

        return plan
