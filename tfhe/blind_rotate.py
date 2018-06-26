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


class BlindRotate(Computation):

    def __init__(self, accum_a, gsw, bara, params: TGswParams):
        self._params = params
        Computation.__init__(self,
            [Parameter('accum_a', Annotation(accum_a, 'io')),
            Parameter('gsw', Annotation(gsw, 'i')),
            Parameter('bara', Annotation(bara, 'i')),
            Parameter('n', Annotation(numpy.int32))])

    def _build_plan(self, plan_factory, device_params, accum_a, gsw, bara, n):
        plan = plan_factory()

        transform = transform_module()

        tlwe_params = self._params.tlwe_params
        k = tlwe_params.k
        l = self._params.l
        N = tlwe_params.N

        batch_shape = accum_a.shape[:-2]

        cdata_forward = plan.persistent_array(transform.cdata_fw)
        cdata_inverse = plan.persistent_array(transform.cdata_inv)

        plan.kernel_call(
            TEMPLATE.get_def("BlindRotate"),
            [accum_a, gsw, bara, cdata_forward, cdata_inverse, n],
            global_size=(helpers.product(batch_shape), transform.threads_per_transform),
            local_size=(1, transform.threads_per_transform),
            render_kwds=dict(
                slices=(len(batch_shape), 1, 1),
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


def BlindRotate_gpu(
        accum: TLweSampleArray, bkFFT: TGswSampleFFTArray, bara, n: int, bk_params: TGswParams):

    print(accum.a.coefsT.shape, bkFFT.samples.a.coefsC.shape, bara.shape, n)
    thr = accum.a.coefsT.thread
    comp = get_computation(thr, BlindRotate, accum.a.coefsT, bkFFT.samples.a.coefsC, bara, bk_params)
    comp(accum.a.coefsT, bkFFT.samples.a.coefsC, bara, n)
