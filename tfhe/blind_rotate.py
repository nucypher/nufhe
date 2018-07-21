import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.cluda import dtypes, functions
import reikna.helpers as helpers

from .gpu_polynomials import TorusPolynomialArray
from .tgsw import TGswParams, TGswSampleArray, TGswSampleFFTArray
from .tlwe import TLweSampleArray
from .polynomial_transform import (
    ForwardTransform, InverseTransform, transformed_dtype, transform_module,
    transformed_internal_dtype, transformed_internal_ctype, transformed_length,
    transformed_mul, transformed_add)
from .computation_cache import get_computation


TEMPLATE = helpers.template_for(__file__)


class BlindRotateKS(Computation):

    def __init__(self, out_a, out_b, accum_a, gsw, ks_a, ks_b, bara, params: TGswParams):
        self._params = params
        Computation.__init__(self,
            [
            Parameter('lwe_a', Annotation(out_a, 'io')),
            Parameter('lwe_b', Annotation(out_b, 'io')),
            Parameter('accum_a', Annotation(accum_a, 'io')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('ks_a', Annotation(ks_a, 'i')),
            Parameter('ks_b', Annotation(ks_b, 'i')),
            Parameter('bara', Annotation(bara, 'i')),
            Parameter('n', Annotation(numpy.int32))])

    def _build_plan(self, plan_factory, device_params, lwe_a, lwe_b, accum_a, gsw, ks_a, ks_b, bara, n):
        plan = plan_factory()

        transform = transform_module()

        tlwe_params = self._params.tlwe_params
        k = tlwe_params.mask_size
        l = self._params.decomp_length
        N = tlwe_params.polynomial_degree

        batch_shape = accum_a.shape[:-2]

        if transform.use_constant_memory:
            cdata_forward = plan.constant_array(transform.cdata_fw)
            cdata_inverse = plan.constant_array(transform.cdata_inv)
        else:
            cdata_forward = plan.persistent_array(transform.cdata_fw)
            cdata_inverse = plan.persistent_array(transform.cdata_inv)

        plan.kernel_call(
            TEMPLATE.get_def("BlindRotateKS"),
            [lwe_a, lwe_b, accum_a, gsw, ks_a, ks_b, bara, cdata_forward, cdata_inverse, n],
            global_size=(helpers.product(batch_shape), transform.threads_per_transform * 4),
            local_size=(1, transform.threads_per_transform * 4),
            render_kwds=dict(
                slices=(len(batch_shape), 1, 1),
                slices2=(len(batch_shape), 1),
                slices3=(len(batch_shape),),
                transform=transform,
                k=k,
                l=l,
                params=self._params,
                mul=transformed_mul(),
                add=transformed_add(),
                tr_ctype=transformed_internal_ctype(),
                )
            )

        return plan


def BlindRotate_ks_gpu(
        lwe_out, accum: TLweSampleArray, bkFFT: TGswSampleFFTArray, ks_a, ks_b, bara, n: int, bk_params: TGswParams):

    thr = accum.a.coefsT.thread
    comp = get_computation(thr, BlindRotateKS,
        lwe_out.a, lwe_out.b, accum.a.coefsT, bkFFT.samples.a.coefsC, ks_a, ks_b, bara, bk_params)
    comp(lwe_out.a, lwe_out.b, accum.a.coefsT, bkFFT.samples.a.coefsC, ks_a, ks_b, bara, n)
