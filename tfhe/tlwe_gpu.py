import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.algorithms import PureParallel
import reikna.helpers as helpers

from .numeric_functions import Torus32, Int32, Float
from .polynomial_transform import get_transform
from .performance import PerformanceParameters, performance_parameters_for_device


TEMPLATE = helpers.template_for(__file__)


class TLweNoiselessTrivial(Computation):

    def __init__(self, params: 'TLweParams', shape):
        a_type = Type(Torus32, shape + (params.mask_size + 1, params.polynomial_degree))
        cv_type = Type(Float, shape + (params.mask_size + 1,))
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

            if (${idxs[-1]} == 0)
            {
                ${current_variances.store_idx}(${", ".join(idxs[:-1])}, 0);
            }
            """,
            render_kwds=dict(mask_size=self._mask_size))

        plan.computation_call(fill, a, current_variances, mu)

        return plan


# TODO: can be made faster by using local memory
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
            global_size=(batch_len, self._mask_size, self._polynomial_degree),
            render_kwds=dict(
                slices=(len(result_b.shape), 1, 1),
                mask_size=self._mask_size,
                polynomial_degree=self._polynomial_degree))

        return plan


class TLweEncryptZero(Computation):

    def __init__(
            self, params: 'TLweParams', shape, noise: float, perf_params: PerformanceParameters):

        polynomial_degree = params.polynomial_degree
        mask_size = params.mask_size

        result_a = Type(Torus32, shape + (mask_size + 1, polynomial_degree))
        result_cv = Type(Float, shape)
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

        perf_params = performance_parameters_for_device(self._perf_params, device_params)

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
            global_size=(batch_len, self._mask_size + 1, polynomial_degree),
            render_kwds=dict(
                noise=self._noise, mask_size=self._mask_size,
                noises1_slices=(len(batch_shape), 1, 1),
                noises2_slices=(len(batch_shape), 1),
                cv_slices=(len(batch_shape),)
                ))

        return plan
