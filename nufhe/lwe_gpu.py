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
from reikna.algorithms import PureParallel, Reduce, predicate_sum
from reikna.cluda import functions, Snippet
import reikna.helpers as helpers
from reikna import transformations

from .numeric_functions import Torus32, Int32, ErrorFloat


TEMPLATE = helpers.template_for(__file__)


class MatrixMulVector(Computation):

    def __init__(self, matrix_t):
        Computation.__init__(self, [
            Parameter('output', Annotation(Type(matrix_t.dtype, matrix_t.shape[:-1]), 'o')),
            Parameter('matrix', Annotation(matrix_t, 'i')),
            Parameter('vector', Annotation(Type(matrix_t.dtype, matrix_t.shape[-1]), 'i'))])

    def _build_plan(self, plan_factory, device_params, output, matrix, vector):
        plan = plan_factory()

        summation = Reduce(matrix, predicate_sum(matrix.dtype), axes=(len(matrix.shape)-1,))

        mul_vec = Transformation([
            Parameter('output', Annotation(matrix, 'o')),
            Parameter('matrix', Annotation(matrix, 'i')),
            Parameter('vector', Annotation(vector, 'i'))],
            """
            ${output.store_same}(${mul}(${matrix.load_same}, ${vector.load_idx}(${idxs[-1]})));
            """,
            render_kwds=dict(mul=functions.mul(matrix.dtype, vector.dtype)),
            connectors=['output', 'matrix'])

        summation.parameter.input.connect(
            mul_vec, mul_vec.output, matrix=mul_vec.matrix, vector=mul_vec.vector)

        plan.computation_call(summation, output, matrix, vector)

        return plan


class MakeLweKeyswitchKey(Computation):

    def __init__(
            self, input_size: int, output_size: int,
            decomp_length: int, log2_base: int, noise: float):

        base = 2**log2_base

        a = Type(Torus32, (input_size, decomp_length, base, output_size))
        b = Type(Torus32, (input_size, decomp_length, base))
        cv = Type(ErrorFloat, (input_size, decomp_length, base))
        in_key = Type(Int32, input_size)
        out_key = Type(Int32, output_size)

        noises_a = Type(Torus32, (input_size, decomp_length, base - 1, output_size))
        noises_b = Type(Torus32, (input_size, decomp_length, base - 1))

        self._output_size = output_size
        self._log2_base = log2_base
        self._noise = noise

        Computation.__init__(self, [
            Parameter('ks_a', Annotation(a, 'o')),
            Parameter('ks_b', Annotation(b, 'o')),
            Parameter('ks_cv', Annotation(cv, 'o')),
            Parameter('in_key', Annotation(in_key, 'i')),
            Parameter('out_key', Annotation(out_key, 'i')),
            Parameter('noises_a', Annotation(noises_a, 'i')),
            Parameter('noises_b', Annotation(noises_b, 'i'))])

    def _build_plan(
            self, plan_factory, device_params,
            ks_a, ks_b, ks_cv, in_key, out_key, noises_a, noises_b):

        plan = plan_factory()

        extracted_n, t, base, inner_n = ks_a.shape

        mul_key = MatrixMulVector(noises_a)
        b_term = plan.temp_array_like(mul_key.parameter.output)

        build_keyswitch = PureParallel([
            Parameter('ks_a', Annotation(ks_a, 'o')),
            Parameter('ks_b', Annotation(ks_b, 'o')),
            Parameter('ks_cv', Annotation(ks_cv, 'o')),
            Parameter('in_key', Annotation(in_key, 'i')),
            Parameter('b_term', Annotation(b_term, 'i')),
            Parameter('noises_a', Annotation(noises_a, 'i')),
            Parameter('noises_b', Annotation(noises_b, 'i'))],
            Snippet(
                TEMPLATE.get_def("make_lwe_keyswitch_key"),
                render_kwds=dict(
                    log2_base=self._log2_base, output_size=self._output_size,
                    noise=self._noise)),
            guiding_array="ks_b")

        plan.computation_call(mul_key, b_term, noises_a, out_key)
        plan.computation_call(
            build_keyswitch,
            ks_a, ks_b, ks_cv, in_key, b_term, noises_a, noises_b)

        return plan


class LweKeyswitch(Computation):

    def __init__(
            self, result_shape_info,
            input_size: int, output_size: int, decomp_length: int, log2_base: int):

        base = 2**log2_base

        a = result_shape_info.a
        b = result_shape_info.b
        cv = result_shape_info.current_variances

        ks_a = Type(Torus32, (input_size, decomp_length, base, output_size))
        ks_b = Type(Torus32, (input_size, decomp_length, base))
        ks_cv = Type(ErrorFloat, (input_size, decomp_length, base))

        source_a = Type(Torus32, result_shape_info.shape + (input_size,))
        source_b = Type(Torus32, result_shape_info.shape)

        self._decomp_length = decomp_length
        self._input_size = input_size
        self._output_size = output_size
        self._log2_base = log2_base

        Computation.__init__(self, [
            Parameter('result_a', Annotation(a, 'io')),
            Parameter('result_b', Annotation(b, 'io')),
            Parameter('result_cv', Annotation(cv, 'io')),
            Parameter('ks_a', Annotation(ks_a, 'i')),
            Parameter('ks_b', Annotation(ks_b, 'i')),
            Parameter('ks_cv', Annotation(ks_cv, 'i')),
            Parameter('source_a', Annotation(source_a, 'i')),
            Parameter('source_b', Annotation(source_b, 'i'))])

    def _build_plan(
            self, plan_factory, device_params,
            result_a, result_b, result_cv,
            ks_a, ks_b, ks_cv, source_a, source_b):

        plan = plan_factory()

        batch_shape = result_a.shape[:-1]

        plan.kernel_call(
            TEMPLATE.get_def("lwe_keyswitch"),
            [result_a, result_b, result_cv, ks_a, ks_b, ks_cv, source_a, source_b],
            kernel_name="lwe_keyswitch",
            global_size=(helpers.product(batch_shape), self._output_size),
            render_kwds=dict(
                slices=(len(batch_shape), 1),
                output_size=self._output_size,
                input_size=self._input_size,
                log2_base=self._log2_base,
                decomp_length=self._decomp_length,
                ))

        return plan


class LweEncrypt(Computation):

    def __init__(self, shape, lwe_size: int, noise: float):

        result_a = Type(Torus32, shape + (lwe_size,))
        result_b = Type(Torus32, shape)
        result_cv = Type(ErrorFloat, shape)
        messages = Type(Torus32, shape)
        key = Type(Int32, (lwe_size,))
        noises_a = Type(Torus32, shape + (lwe_size,))
        noises_b = Type(Torus32, shape)

        self._noise = noise

        Computation.__init__(self, [
            Parameter('result_a', Annotation(result_a, 'o')),
            Parameter('result_b', Annotation(result_b, 'o')),
            Parameter('result_cv', Annotation(result_cv, 'o')),
            Parameter('messages', Annotation(messages, 'i')),
            Parameter('key', Annotation(key, 'i')),
            Parameter('noises_a', Annotation(noises_a, 'i')),
            Parameter('noises_b', Annotation(noises_b, 'i'))])

    def _build_plan(
            self, plan_factory, device_params,
            result_a, result_b, result_cv, messages, key, noises_a, noises_b):

        plan = plan_factory()

        mul_key = MatrixMulVector(noises_a)

        fill_b_cv = Transformation([
            Parameter('result_b', Annotation(result_b, 'o')),
            Parameter('result_cv', Annotation(result_cv, 'o')),
            Parameter('messages', Annotation(messages, 'i')),
            Parameter('noises_a_times_key', Annotation(noises_b, 'i')),
            Parameter('noises_b', Annotation(noises_b, 'i'))],
            """
            ${result_b.store_same}(
                ${noises_b.load_same}
                + ${messages.load_same}
                + ${noises_a_times_key.load_same});
            ${result_cv.store_same}(${noise**2});
            """,
            connectors=['noises_a_times_key'],
            render_kwds=dict(noise=self._noise))

        mul_key.parameter.output.connect(
            fill_b_cv, fill_b_cv.noises_a_times_key,
            b=fill_b_cv.result_b, cv=fill_b_cv.result_cv, messages=fill_b_cv.messages,
            noises_b=fill_b_cv.noises_b)

        plan.computation_call(mul_key, result_b, result_cv, messages, noises_b, noises_a, key)
        plan.computation_call(
            PureParallel.from_trf(transformations.copy(noises_a)),
            result_a, noises_a)

        return plan


class LweDecrypt(Computation):
    """
    Compute the phase of the sample using the secret key: phi = b - a.s
    """

    def __init__(self, shape, lwe_size):

        a = Type(Torus32, shape + (lwe_size,))
        b = Type(Torus32, shape)
        key = Type(Int32, (lwe_size,))

        Computation.__init__(self, [
            Parameter('result', Annotation(b, 'o')),
            Parameter('lwe_a', Annotation(a, 'i')),
            Parameter('lwe_b', Annotation(b, 'i')),
            Parameter('key', Annotation(key, 'i'))])

    def _build_plan(self, plan_factory, device_params, result, lwe_a, lwe_b, key):

        plan = plan_factory()

        mul_key = MatrixMulVector(lwe_a)

        fill_res = Transformation([
            Parameter('result', Annotation(result, 'o')),
            Parameter('b', Annotation(lwe_b, 'i')),
            Parameter('a_times_key', Annotation(lwe_b, 'i'))],
            """
            ${result.store_same}(${b.load_same} - ${a_times_key.load_same});
            """,
            connectors=['a_times_key'])

        mul_key.parameter.output.connect(
            fill_res, fill_res.a_times_key,
            result=fill_res.result, b=fill_res.b)

        plan.computation_call(mul_key, result, lwe_b, lwe_a, key)

        return plan


class LweLinear(Computation):

    def __init__(self, result_shape_info, source_shape_info, add_result=False):

        self._add_result = add_result

        Computation.__init__(self, [
            Parameter('result_a', Annotation(result_shape_info.a, 'o')),
            Parameter('result_b', Annotation(result_shape_info.b, 'o')),
            Parameter('result_cv', Annotation(result_shape_info.current_variances, 'o')),
            Parameter('source_a', Annotation(source_shape_info.a, 'i')),
            Parameter('source_b', Annotation(source_shape_info.b, 'i')),
            Parameter('source_cv', Annotation(source_shape_info.current_variances, 'i')),
            Parameter('coeff', Annotation(Type(Torus32)))])

    def _build_plan(
            self, plan_factory, device_params,
            result_a, result_b, result_cv, source_a, source_b, source_cv, coeff):

        plan = plan_factory()
        plan.kernel_call(
            TEMPLATE.get_def("lwe_linear"),
            [result_a, result_b, result_cv, source_a, source_b, source_cv, coeff],
            kernel_name="lwe_linear",
            global_size=result_a.shape,
            render_kwds=dict(
                add_result=self._add_result,
                ))

        return plan


class LweNoiselessTrivial(Computation):

    def __init__(self, result_shape_info, source_shape):
        Computation.__init__(self, [
            Parameter('result_a', Annotation(result_shape_info.a, 'o')),
            Parameter('result_b', Annotation(result_shape_info.b, 'o')),
            Parameter('result_cv', Annotation(result_shape_info.current_variances, 'o')),
            Parameter('mus', Annotation(Type(Torus32, source_shape), 'i'))])

    def _build_plan(self, plan_factory, device_params, result_a, result_b, result_cv, mus):

        plan = plan_factory()
        plan.kernel_call(
            TEMPLATE.get_def("lwe_noiseless_trivial"),
            [result_a, result_b, result_cv, mus],
            kernel_name="lwe_noiseless_trivial",
            global_size=result_a.shape)

        return plan


def LweNoiselessTrivialConstant(shape_info):
    comp = LweNoiselessTrivial(shape_info, shape_info.shape)
    bc = transformations.broadcast_param(comp.parameter.mus)
    comp.parameter.mus.connect(bc, bc.output, mu=bc.param)
    return comp
