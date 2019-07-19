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
from reikna.algorithms import PureParallel
import reikna.helpers as helpers

from .numeric_functions import Torus32, Int32, ErrorFloat
from .polynomial_transform import get_transform
from .performance import PerformanceParametersForDevice


TEMPLATE = helpers.template_for(__file__)


class TLweNoiselessTrivial(Computation):

    def __init__(self, params: 'TLweParams', shape):
        a_type = Type(Torus32, shape + (params.mask_size + 1, params.polynomial_degree))
        cv_type = Type(ErrorFloat, shape)
        mu_type = Type(Torus32, shape + (params.polynomial_degree,))

        self._mask_size = params.mask_size

        Computation.__init__(self,
            [Parameter('a', Annotation(a_type, 'o')),
            Parameter('current_variances', Annotation(cv_type, 'o')),
            Parameter('mu', Annotation(mu_type, 'i'))])

    def _build_plan(self, plan_factory, device_params, a, current_variances, mu):
        plan = plan_factory()

        fill = PureParallel([
            Parameter('a', Annotation(a, 'o')),
            Parameter('current_variances', Annotation(current_variances, 'o')),
            Parameter('mu', Annotation(mu, 'i'))],
            """
            ${a.ctype} a;
            if (${idxs[-2]} == ${mask_size})
            {
                a = ${mu.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-1]});
            }
            else
            {
                a = 0;
            }
            ${a.store_same}(a);

            if (${idxs[-1]} == 0 && ${idxs[-2]} == 0)
            {
                ${current_variances.store_idx}(${", ".join(idxs[:-2])}, 0);
            }
            """,
            render_kwds=dict(mask_size=self._mask_size))

        plan.computation_call(fill, a, current_variances, mu)

        return plan


class TLweExtractLweSamples(Computation):

    def __init__(self, params: 'TLweParams', shape):

        self._mask_size = params.mask_size
        self._polynomial_degree = params.polynomial_degree

        result_a = Type(Torus32, shape + (params.extracted_lweparams.size,))
        result_b = Type(Torus32, shape)
        tlwe_a = Type(Torus32, shape + (params.mask_size + 1, params.polynomial_degree))

        Computation.__init__(self, [
            Parameter('result_a', Annotation(result_a, 'o')),
            Parameter('result_b', Annotation(result_b, 'o')),
            Parameter('tlwe_a', Annotation(tlwe_a, 'i'))])

    def _build_plan(self, plan_factory, device_params, result_a, result_b, tlwe_a):
        plan = plan_factory()

        batch_len = helpers.product(result_b.shape)

        plan.kernel_call(
            TEMPLATE.get_def('tlwe_extract_lwe_samples'),
            [result_a, result_b, tlwe_a],
            kernel_name="tlwe_extract_lwe_samples",
            global_size=(batch_len, self._mask_size, self._polynomial_degree),
            render_kwds=dict(
                slices=(len(result_b.shape), 1, 1),
                mask_size=self._mask_size,
                polynomial_degree=self._polynomial_degree))

        return plan


class TLweEncryptZero(Computation):

    def __init__(
            self, params: 'TLweParams', shape, noise: float,
            perf_params: PerformanceParametersForDevice):

        polynomial_degree = params.polynomial_degree
        mask_size = params.mask_size

        result_a = Type(Torus32, shape + (mask_size + 1, polynomial_degree))
        result_cv = Type(ErrorFloat, shape)
        key = Type(Int32, (mask_size, polynomial_degree))
        noises1 = Type(Torus32, shape + (mask_size, polynomial_degree))
        noises2 = Type(Torus32, shape + (polynomial_degree,))

        self._transform_type = params.transform_type
        self._noise = noise
        self._mask_size = mask_size
        self._polynomial_degree = polynomial_degree
        self._perf_params = perf_params

        Computation.__init__(self, [
            Parameter('result_a', Annotation(result_a, 'o')),
            Parameter('result_cv', Annotation(result_cv, 'o')),
            Parameter('key', Annotation(key, 'i')),
            Parameter('noises1', Annotation(noises1, 'i')),
            Parameter('noises2', Annotation(noises2, 'i'))])

    def _build_plan(
            self, plan_factory, device_params,
            result_a, result_cv, key, noises1, noises2):

        plan = plan_factory()

        polynomial_degree = self._polynomial_degree
        batch_shape = result_a.shape[:-2]
        batch_len = helpers.product(batch_shape)

        perf_params = self._perf_params

        transform = get_transform(self._transform_type)

        ft_key = transform.ForwardTransform(key.shape[:-1], polynomial_degree, perf_params)
        key_tr = plan.temp_array_like(ft_key.parameter.output)

        ft_noises = transform.ForwardTransform(noises1.shape[:-1], polynomial_degree, perf_params)
        noises1_tr = plan.temp_array_like(ft_noises.parameter.output)

        ift = transform.InverseTransform(noises1.shape[:-1], polynomial_degree, perf_params)
        ift_res = plan.temp_array_like(ift.parameter.output)

        mul_tr = Transformation(
            [
                Parameter('output', Annotation(ift.parameter.input, 'o')),
                Parameter('key', Annotation(key_tr, 'i')),
                Parameter('noises1', Annotation(noises1_tr, 'i'))
            ],
            """
            ${output.store_same}(${tr_ctype}unpack(${mul}(
                ${tr_ctype}pack(${key.load_idx}(${idxs[-2]}, ${idxs[-1]})),
                ${tr_ctype}pack(${noises1.load_same})
                )));
            """,
            connectors=['output', 'noises1'],
            render_kwds=dict(
                mul=transform.transformed_mul(perf_params),
                tr_ctype=transform.transformed_internal_ctype()))

        ift.parameter.input.connect(mul_tr, mul_tr.output, key=mul_tr.key, noises1=mul_tr.noises1)

        plan.computation_call(ft_key, key_tr, key)
        plan.computation_call(ft_noises, noises1_tr, noises1)
        plan.computation_call(ift, ift_res, key_tr, noises1_tr)
        plan.kernel_call(
            TEMPLATE.get_def("tlwe_encrypt_zero_fill_result"),
            [result_a, result_cv, noises1, noises2, ift_res],
            kernel_name="tlwe_encrypt_zero_fill_result",
            global_size=(batch_len, self._mask_size + 1, polynomial_degree),
            render_kwds=dict(
                noise=self._noise, mask_size=self._mask_size,
                noises1_slices=(len(batch_shape), 1, 1),
                noises2_slices=(len(batch_shape), 1),
                cv_slices=(len(batch_shape),)
                ))

        return plan


class TLweTransformSamples(Computation):
    """
    Convert given Torus32 values to the transformed space and prepare them for fast multiplication
    (converting to Montgomery for NTT and doing nothing for FFT).
    """

    def __init__(
            self, params: 'TLweParams', shape, perf_params: PerformanceParametersForDevice):

        self._transform_type = params.transform_type
        self._perf_params = perf_params

        batch_shape = shape[:-1]
        polynomial_degree = params.polynomial_degree
        transform = get_transform(self._transform_type)
        tlength = transform.transformed_length(polynomial_degree)
        tdtype = transform.transformed_dtype()

        prepared_values = Type(tdtype, batch_shape + (tlength,))
        values = Type(Torus32, shape)

        Computation.__init__(self, [
            Parameter('prepared_samples', Annotation(prepared_values, 'o')),
            Parameter('values', Annotation(values, 'i'))])

    def _build_plan(self, plan_factory, device_params, prepared_values, values):

        plan = plan_factory()

        transform = get_transform(self._transform_type)

        comp = transform.ForwardTransform(values.shape[:-1], values.shape[-1], self._perf_params)
        prepare = transform.get_prepare_for_mul_trf(prepared_values.shape)
        comp.parameter.output.connect(prepare, prepare.input, prepared=prepare.output)

        plan.computation_call(comp, prepared_values, values)

        return plan
