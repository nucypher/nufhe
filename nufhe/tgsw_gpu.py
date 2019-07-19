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

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
import reikna.helpers as helpers

from .numeric_functions import Torus32, Int32
from .polynomial_transform import get_transform
from .performance import PerformanceParametersForDevice


TEMPLATE = helpers.template_for(__file__)


def get_tgsw_polynomial_decomp_trf(params: 'TGswParams', shape):
    tlwe_params = params.tlwe_params
    decomp_length = params.decomp_length
    mask_size = tlwe_params.mask_size
    polynomial_degree = tlwe_params.polynomial_degree

    result = Type(Int32, shape + (mask_size + 1, decomp_length, polynomial_degree))
    sample = Type(Torus32, shape + (mask_size + 1, polynomial_degree))
    return Transformation([
        Parameter('result', Annotation(result, 'o')),
        Parameter('sample', Annotation(sample, 'i'))],
        """
        <%
            mask = 2**params.bs_log2_base - 1
            half_base = 2**(params.bs_log2_base - 1)
        %>
        ${sample.ctype} sample = ${sample.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-1]});
        int decomp_shift = 32 - (${idxs[-2]} + 1) * ${params.bs_log2_base};
        ${result.store_same}(
            (((sample + (${params.offset})) >> decomp_shift) & ${mask}) - ${half_base}
        );
        """,
        connectors=['results'],
        render_kwds=dict(params=params))


# result = result + p*sample
def get_tlwe_transformed_add_mul_to_trf(
        params: 'TGswParams', shape, bk_len: int, perf_params: PerformanceParametersForDevice):

    tlwe_params = params.tlwe_params
    decomp_length = params.decomp_length
    mask_size = tlwe_params.mask_size
    polynomial_degree = tlwe_params.polynomial_degree

    transform = get_transform(params.tlwe_params.transform_type)
    tdtype = transform.transformed_dtype()
    tlength = transform.transformed_length(polynomial_degree)
    tr_ctype = transform.transformed_internal_ctype()

    result = Type(tdtype, shape + (mask_size + 1, tlength))
    sample = Type(tdtype, shape + (mask_size + 1, decomp_length, tlength))
    bootstrap_key = Type(tdtype, (bk_len, mask_size + 1, decomp_length, mask_size + 1, tlength))

    return Transformation([
        Parameter('result', Annotation(result, 'o')),
        Parameter('sample', Annotation(sample, 'i')),
        Parameter('bootstrap_key', Annotation(bootstrap_key, 'i')),
        Parameter('bk_row_idx', Annotation(numpy.int32))],
        """
        ${tr_ctype} result = ${tr_ctype}pack(${dtypes.c_constant(0, result.dtype)});

        %for mask_idx in range(mask_size + 1):
        %for decomp_idx in range(decomp_length):
        {
            ${tr_ctype} a = ${tr_ctype}pack(
                ${sample.load_idx}(
                    ${", ".join(idxs[:-2])}, ${mask_idx}, ${decomp_idx}, ${idxs[-1]})
                );
            ${tr_ctype} b = ${tr_ctype}pack(
                ${bootstrap_key.load_idx}(
                    ${bk_row_idx}, ${mask_idx}, ${decomp_idx}, ${idxs[-2]}, ${idxs[-1]})
                );
            result = ${add}(result, ${mul_prepared}(a, b));
        }
        %endfor
        %endfor

        ${result.store_same}(${tr_ctype}unpack(result));
        """,
        connectors=['result'],
        render_kwds=dict(
            mask_size=mask_size,
            decomp_length=decomp_length,
            add=transform.transformed_add(perf_params),
            mul_prepared=transform.transformed_mul_prepared(perf_params),
            tr_ctype=tr_ctype))


class TGswTransformedExternalMul(Computation):

    def __init__(
            self, params: 'TGswParams', shape, bk_len, perf_params: PerformanceParametersForDevice):

        mask_size = params.tlwe_params.mask_size
        polynomial_degree = params.tlwe_params.polynomial_degree
        decomp_length = params.decomp_length

        transform = get_transform(params.tlwe_params.transform_type)
        tdtype = transform.transformed_dtype()
        tlength = transform.transformed_length(polynomial_degree)

        accum = Type(Torus32, shape + (mask_size + 1, polynomial_degree))
        bootstrap_key = Type(tdtype, (bk_len, mask_size + 1, decomp_length, mask_size + 1, tlength))

        self._params = params
        self._perf_params = perf_params
        self._shape = shape
        self._bk_len = bk_len

        Computation.__init__(self,
            [Parameter('accum', Annotation(accum, 'io')),
            Parameter('bootstrap_key', Annotation(bootstrap_key, 'i')),
            Parameter('bk_row_idx', Annotation(numpy.int32))])

    def _build_plan(self, plan_factory, device_params, accum, bootstrap_key, bk_row_idx):
        plan = plan_factory()

        perf_params = self._perf_params

        params = self._params

        tlwe_params = params.tlwe_params
        polynomial_degree = tlwe_params.polynomial_degree

        batch_shape = self._shape

        transform = get_transform(tlwe_params.transform_type)

        decomp = get_tgsw_polynomial_decomp_trf(params, batch_shape)
        decomp_and_ftr = transform.ForwardTransform(
            decomp.result.shape[:-1], polynomial_degree, perf_params)
        decomp_and_ftr.parameter.input.connect(decomp, decomp.result, sample=decomp.sample)

        add_mul = get_tlwe_transformed_add_mul_to_trf(
            params, batch_shape, self._bk_len, perf_params)
        add_mul_and_itr = transform.InverseTransform(
            add_mul.result.shape[:-1], polynomial_degree, perf_params)
        add_mul_and_itr.parameter.input.connect(
            add_mul, add_mul.result,
            tr_sample=add_mul.sample, bootstrap_key=add_mul.bootstrap_key,
            bk_row_idx=add_mul.bk_row_idx)

        tr_sample = plan.temp_array_like(decomp_and_ftr.parameter.output)

        plan.computation_call(decomp_and_ftr, tr_sample, accum)
        plan.computation_call(add_mul_and_itr, accum, tr_sample, bootstrap_key, bk_row_idx)

        return plan


class TGswAddMessage(Computation):

    def __init__(self, params: 'TGswParams', shape):

        self._params = params

        decomp_length = params.decomp_length
        mask_size = params.tlwe_params.mask_size
        polynomial_degree = params.tlwe_params.polynomial_degree

        result_a = Type(
            Torus32, shape + (mask_size + 1, decomp_length, mask_size + 1, polynomial_degree))
        messages = Type(Torus32, shape)

        Computation.__init__(self,
            [Parameter('result_a', Annotation(result_a, 'o')),
            Parameter('messages', Annotation(messages, 'i'))])

    def _build_plan(self, plan_factory, device_params, result_a, messages):
        plan = plan_factory()

        batch_len = helpers.product(messages.shape)
        plan.kernel_call(
            TEMPLATE.get_def("tgsw_add_message"),
            [result_a, messages],
            kernel_name="tgsw_add_message",
            global_size=(batch_len,),
            render_kwds=dict(
                slices=(len(messages.shape), 1, 1, 1, 1),
                base_powers=self._params.base_powers,
                decomp_length=self._params.decomp_length,
                mask_size=self._params.tlwe_params.mask_size))

        return plan
