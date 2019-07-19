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

from reikna.core import Computation, Parameter, Annotation, Type
from reikna.cluda import OutOfResourcesError, ocl_id
import reikna.cluda.dtypes as dtypes
import reikna.helpers as helpers

from .lwe import LweParams, LweSampleArray, LweKeyswitch
from .tgsw import TGswParams, TransformedTGswSampleArray
from .tlwe import TLweSampleArray
from .computation_cache import get_computation
from .polynomial_transform import get_transform
from .performance import PerformanceParametersForDevice
from .numeric_functions import Torus32, ErrorFloat


TEMPLATE = helpers.template_for(__file__)


def single_kernel_bootstrap_supported(nufhe_params, device_params, raise_exception=False):

    if device_params.api_id == ocl_id():
        # OpenCL uses some local memory for kernel arguments if there are many of them,
        # and we need all the available local memory for internal buffers.
        if raise_exception:
            raise ValueError("Single-kernel bootstrap is not supported for OpenCL")
        else:
            return False

    transform_type = nufhe_params.tgsw_params.tlwe_params.transform_type
    reqs = get_transform(transform_type).transform_module_requirements()

    mask_size = nufhe_params.tgsw_params.tlwe_params.mask_size
    decomp_length = nufhe_params.tgsw_params.decomp_length

    if not (mask_size == 1 and decomp_length == 2):
        if raise_exception:
            raise ValueError(
                "Single-kernel bootstrap is only supported for mask_size=1 and decomp_length=2")
        else:
            return False

    skb_transforms = (mask_size + 1) * decomp_length
    threads_per_transform = reqs['threads_per_transform']
    max_work_group_size = device_params.max_work_group_size
    if not threads_per_transform * skb_transforms <= max_work_group_size:
        if raise_exception:
            raise ValueError(
                "The chosen device does not support a block/workgroup size big enough "
                "to run single-kernel bootstrap")
        else:
            return False

    tr_size = reqs['transform_length'] * reqs['elem_dtype_itemsize']
    temp_size = reqs['temp_length'] * reqs['temp_dtype_itemsize']
    poly_dtype_itemsize = dtypes.normalize_type(Torus32).itemsize
    sh_size = max(tr_size, temp_size)
    required_lmem_size = (
        sh_size * ((mask_size + 1) * decomp_length + mask_size)
        + (mask_size + 1) * reqs['polynomial_length'] * poly_dtype_itemsize)
    if required_lmem_size > device_params.local_mem_size:
        if raise_exception:
            raise ValueError(
                "The chosen device does not have enough shared/local memory "
                "to run single-kernel bootstrap")
        else:
            return False

    return True


class BlindRotate(Computation):

    def __init__(
            self, params: TGswParams, in_out_params: LweParams, shape,
            perf_params: PerformanceParametersForDevice):

        tlwe_params = params.tlwe_params
        decomp_length = params.decomp_length
        mask_size = tlwe_params.mask_size
        polynomial_degree = tlwe_params.polynomial_degree
        input_size = params.tlwe_params.extracted_lweparams.size
        output_size = in_out_params.size

        assert mask_size == 1 and decomp_length == 2

        transform_type = params.tlwe_params.transform_type
        transform = get_transform(transform_type)
        tlength = transform.transformed_length(polynomial_degree)
        tdtype = transform.transformed_dtype()

        out_a = Type(Torus32, shape + (input_size,))
        out_b = Type(Torus32, shape)
        accum_a = Type(Torus32, shape + (mask_size + 1, polynomial_degree))
        gsw = Type(tdtype, (output_size, mask_size + 1, decomp_length, mask_size + 1, tlength))
        bara = Type(Torus32, shape + (output_size,))

        self._params = params
        self._in_out_params = in_out_params
        self._perf_params = perf_params

        Computation.__init__(self,
            [
            Parameter('lwe_a', Annotation(out_a, 'io')),
            Parameter('lwe_b', Annotation(out_b, 'io')),
            Parameter('accum_a', Annotation(accum_a, 'io')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('bara', Annotation(bara, 'i'))])

    def _build_plan(self, plan_factory, device_params, lwe_a, lwe_b, accum_a, gsw, bara):

        params = self._params
        tlwe_params = params.tlwe_params
        decomp_length = params.decomp_length
        mask_size = tlwe_params.mask_size

        perf_params = self._perf_params
        transform_type = self._params.tlwe_params.transform_type
        transform = get_transform(transform_type)

        transform_module = transform.transform_module(perf_params, multi_iter=True)

        batch_shape = accum_a.shape[:-2]

        min_local_size = decomp_length * (mask_size + 1) * transform_module.threads_per_transform
        local_size = device_params.max_work_group_size
        while local_size >= min_local_size:

            plan = plan_factory()

            if transform_module.use_constant_memory:
                cdata_forward = plan.constant_array(transform_module.cdata_fw)
                cdata_inverse = plan.constant_array(transform_module.cdata_inv)
            else:
                cdata_forward = plan.persistent_array(transform_module.cdata_fw)
                cdata_inverse = plan.persistent_array(transform_module.cdata_inv)

            try:
                plan.kernel_call(
                    TEMPLATE.get_def("blind_rotate"),
                    [lwe_a, lwe_b, accum_a, gsw, bara, cdata_forward, cdata_inverse],
                    kernel_name="blind_rotate",
                    global_size=(
                        helpers.product(batch_shape),
                        local_size),
                    local_size=(1, local_size),
                    render_kwds=dict(
                        local_size=local_size,
                        slices=(len(batch_shape), 1, 1),
                        slices2=(len(batch_shape), 1),
                        slices3=(len(batch_shape),),
                        transform=transform_module,
                        mask_size=mask_size,
                        decomp_length=decomp_length,
                        output_size=self._in_out_params.size,
                        input_size=tlwe_params.extracted_lweparams.size,
                        bs_log2_base=self._params.bs_log2_base,
                        mul_prepared=transform.transformed_mul_prepared(perf_params),
                        add=transform.transformed_add(perf_params),
                        tr_ctype=transform.transformed_internal_ctype(),
                        min_blocks=helpers.min_blocks,
                        )
                    )
            except OutOfResourcesError:
                local_size -= transform_module.threads_per_transform
                continue

            return plan

        raise ValueError("Could not find suitable local size for the kernel")


class BlindRotateAndKeySwitch(Computation):

    def __init__(
            self, params: TGswParams, in_out_params: LweParams, result_shape_info,
            ks_log2_base, ks_decomp_length, perf_params: PerformanceParametersForDevice):

        tlwe_params = params.tlwe_params
        bk_decomp_length = params.decomp_length
        mask_size = tlwe_params.mask_size
        polynomial_degree = tlwe_params.polynomial_degree
        input_size = params.tlwe_params.extracted_lweparams.size
        output_size = in_out_params.size
        ks_base = 2**ks_log2_base
        shape = result_shape_info.shape

        transform_type = params.tlwe_params.transform_type
        transform = get_transform(transform_type)
        tlength = transform.transformed_length(polynomial_degree)
        tdtype = transform.transformed_dtype()

        out_a = result_shape_info.a
        out_b = result_shape_info.b
        accum_a = Type(Torus32, shape + (mask_size + 1, polynomial_degree))
        gsw = Type(tdtype, (output_size, mask_size + 1, bk_decomp_length, mask_size + 1, tlength))
        ks_a = Type(Torus32, (input_size, ks_decomp_length, ks_base, output_size))
        ks_b = Type(Torus32, (input_size, ks_decomp_length, ks_base))
        ks_cv = Type(ErrorFloat, (input_size, ks_decomp_length, ks_base))
        bara = Type(Torus32, shape + (output_size,))

        self._result_shape_info = result_shape_info
        self._params = params
        self._in_out_params = in_out_params
        self._ks_log2_base = ks_log2_base
        self._ks_decomp_length = ks_decomp_length
        self._perf_params = perf_params

        Computation.__init__(self,
            [
            Parameter('lwe_a', Annotation(out_a, 'io')),
            Parameter('lwe_b', Annotation(out_b, 'io')),
            Parameter('accum_a', Annotation(accum_a, 'io')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('ks_a', Annotation(ks_a, 'i')),
            Parameter('ks_b', Annotation(ks_b, 'i')),
            Parameter('bara', Annotation(bara, 'i'))])

    def _build_plan(self, plan_factory, device_params, lwe_a, lwe_b, accum_a, gsw, ks_a, ks_b, bara):
        plan = plan_factory()

        batch_shape = accum_a.shape[:-2]
        tlwe_params = self._params.tlwe_params
        ks_decomp_length = self._ks_decomp_length
        input_size = self._params.tlwe_params.extracted_lweparams.size
        output_size = self._in_out_params.size

        extracted_a = plan.temp_array(batch_shape + (input_size,), Torus32)
        extracted_b = plan.temp_array(batch_shape, Torus32)

        blind_rotate = BlindRotate(
            self._params, self._in_out_params, batch_shape, self._perf_params)
        plan.computation_call(blind_rotate, extracted_a, extracted_b, accum_a, gsw, bara)

        ks = LweKeyswitch(
            self._result_shape_info, input_size, output_size, ks_decomp_length, self._ks_log2_base)
        # TODO: need to output current variances properly
        result_cv = plan.temp_array_like(ks.parameter.result_cv)
        ks_cv = plan.temp_array_like(ks.parameter.ks_cv)
        plan.computation_call(ks, lwe_a, lwe_b, result_cv, ks_a, ks_b, ks_cv, extracted_a, extracted_b)

        return plan


def BlindRotate_gpu(
        lwe_out: LweSampleArray, accum: TLweSampleArray,
        bk: 'BootstrapKey', ks: 'LweKeyswitchKey',
        bara, perf_params: PerformanceParametersForDevice, no_keyswitch=False):

    thr = accum.a.coeffs.thread

    shape = lwe_out.shape_info.shape

    if no_keyswitch:
        comp = get_computation(thr, BlindRotate, bk.bk_params, bk.in_out_params, shape, perf_params)
        comp(lwe_out.a, lwe_out.b, accum.a.coeffs, bk.tgsw.samples.a.coeffs, bara)
    else:
        comp = get_computation(
            thr, BlindRotateAndKeySwitch,
            bk.bk_params, bk.in_out_params, lwe_out.shape_info,
            ks.log2_base, ks.decomp_length, perf_params)
        comp(
            lwe_out.a, lwe_out.b, accum.a.coeffs, bk.tgsw.samples.a.coeffs,
            ks.lwe.a, ks.lwe.b, bara)
