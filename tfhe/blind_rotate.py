import numpy

from reikna.core import Computation, Parameter, Annotation
import reikna.helpers as helpers
from reikna.core import Type

from .lwe import LweParams, LweSampleArray
from .tgsw import TGswParams, TGswSampleFFTArray
from .tlwe import TLweSampleArray
from .computation_cache import get_computation
from .polynomial_transform import get_transform
from .lwe import Keyswitch
from .performance import PerformanceParameters, performance_parameters_for_device
from .numeric_functions import Torus32


TEMPLATE = helpers.template_for(__file__)


class BlindRotate(Computation):

    def __init__(
            self, out_a, out_b, accum_a, gsw, bara, params: TGswParams, in_out_params: LweParams,
            perf_params: PerformanceParameters):

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
        plan = plan_factory()

        perf_params = performance_parameters_for_device(self._perf_params, device_params)

        transform_type = self._params.tlwe_params.transform_type
        transform = get_transform(transform_type)

        transform_module = transform.transform_module(perf_params, multi_iter=True)

        tlwe_params = self._params.tlwe_params
        k = tlwe_params.mask_size
        l = self._params.decomp_length

        assert k == 1 and l == 2

        batch_shape = accum_a.shape[:-2]

        if transform_module.use_constant_memory:
            cdata_forward = plan.constant_array(transform_module.cdata_fw)
            cdata_inverse = plan.constant_array(transform_module.cdata_inv)
        else:
            cdata_forward = plan.persistent_array(transform_module.cdata_fw)
            cdata_inverse = plan.persistent_array(transform_module.cdata_inv)

        plan.kernel_call(
            TEMPLATE.get_def("BlindRotate"),
            [lwe_a, lwe_b, accum_a, gsw, bara, cdata_forward, cdata_inverse],
            global_size=(helpers.product(batch_shape), transform_module.threads_per_transform * 4),
            local_size=(1, transform_module.threads_per_transform * 4),
            render_kwds=dict(
                slices=(len(batch_shape), 1, 1),
                slices2=(len(batch_shape), 1),
                slices3=(len(batch_shape),),
                transform=transform_module,
                k=k,
                l=l,
                n=self._in_out_params.size,
                params=self._params,
                mul=transform.transformed_mul(perf_params),
                add=transform.transformed_add(perf_params),
                tr_ctype=transform.transformed_internal_ctype(),
                )
            )

        return plan


class BlindRotateAndKeySwitch(Computation):

    def __init__(
            self, result_shape_info, out_a, out_b, accum_a, gsw, ks_a, ks_b, bara,
            params: TGswParams, in_out_params: LweParams, ks: 'LweKeyswitchKey',
            perf_params: PerformanceParameters):

        out_a = result_shape_info.a
        out_b = result_shape_info.b
        self._result_shape_info = result_shape_info

        self._params = params
        self._in_out_params = in_out_params
        self._ks = ks
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

        tgsw_params = self._params
        eparams = tgsw_params.tlwe_params.extracted_lweparams
        extracted_a = plan.temp_array(batch_shape + (eparams.size,), numpy.int32)
        extracted_b = plan.temp_array(batch_shape, numpy.int32)

        blind_rotate = BlindRotate(
            extracted_a, extracted_b, accum_a, gsw, bara, self._params, self._in_out_params,
            self._perf_params)
        plan.computation_call(blind_rotate, extracted_a, extracted_b, accum_a, gsw, bara)

        outer_n = tgsw_params.tlwe_params.extracted_lweparams.size
        inner_n = self._in_out_params.size
        outer_n = self._ks.input_size
        basebit = self._ks.log2_base
        t = self._ks.decomp_length

        ks = Keyswitch(self._result_shape_info, outer_n, inner_n, t, basebit)
        result_cv = plan.temp_array_like(ks.parameter.result_cv)
        ks_cv = plan.temp_array_like(ks.parameter.ks_cv)
        plan.computation_call(ks, lwe_a, lwe_b, result_cv, ks_a, ks_b, ks_cv, extracted_a, extracted_b)

        return plan



def BlindRotate_gpu(
        lwe_out: LweSampleArray, accum: TLweSampleArray,
        bk: 'LweBootstrappingKeyFFT', bara, perf_params: PerformanceParameters, no_keyswitch=False):

    thr = accum.a.coefsT.thread

    if no_keyswitch:
        comp = get_computation(thr, BlindRotate,
            lwe_out.a, lwe_out.b, accum.a.coefsT,
            bk.bkFFT.samples.a.coefsC, bara, bk.bk_params, bk.in_out_params, perf_params)
        comp(lwe_out.a, lwe_out.b, accum.a.coefsT, bk.bkFFT.samples.a.coefsC, bara)
    else:
        comp = get_computation(thr, BlindRotateAndKeySwitch,
            lwe_out.shape_info, lwe_out.a, lwe_out.b, accum.a.coefsT,
            bk.bkFFT.samples.a.coefsC, bk.ks.lwe.a, bk.ks.lwe.b, bara,
            bk.bk_params, bk.in_out_params, bk.ks, perf_params)
        comp(
            lwe_out.a, lwe_out.b, accum.a.coefsT, bk.bkFFT.samples.a.coefsC,
            bk.ks.lwe.a, bk.ks.lwe.b, bara)

