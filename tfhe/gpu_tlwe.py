import numpy

from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.algorithms import PureParallel
import reikna.transformations as transformations
from reikna.cluda import dtypes, functions

from .tlwe import TLweSampleArray, TLweParams
from .polynomials import TorusPolynomialArray
from .lwe import LweSampleArray, LweParams
from .computation_cache import get_computation

from .gpu_polynomials import *


class TLweNoiselessTrivial(Computation):

    def __init__(self, a):

        assert len(a.shape) == 3

        cv_type = Type(numpy.float64, a.shape[:-1])
        mu_type = Type(a.dtype, (a.shape[0], a.shape[-1]))

        self._fill_a = PureParallel(
            [Parameter('a', Annotation(a, 'o')),
            Parameter('mu', Annotation(mu_type, 'i'))],
            """
            ${a.ctype} a;
            if (${idxs[1]} == ${k})
            {
                a = ${mu.load_idx}(${idxs[0]}, ${idxs[2]});
            }
            else
            {
                a = 0;
            }
            ${a.store_same}(a);
            """,
            render_kwds=dict(k=a.shape[1] - 1))

        self._fill_cv = PureParallel(
            [Parameter('current_variances', Annotation(cv_type, 'o'))],
            """
            ${current_variances.store_same}(0);
            """)

        Computation.__init__(self,
            [Parameter('a', Annotation(a, 'o')),
            Parameter('current_variances', Annotation(cv_type, 'o')),
            Parameter('mu', Annotation(mu_type, 'i'))])

    def _build_plan(self, plan_factory, device_params, a, current_variances, mu):
        plan = plan_factory()
        plan.computation_call(self._fill_a, a, mu)
        plan.computation_call(self._fill_cv, current_variances)
        return plan


def tLweNoiselessTrivial_gpu(result: TLweSampleArray, mu: TorusPolynomialArray, params: TLweParams):
    thr = result.a.coefsT.thread
    comp = get_computation(thr, TLweNoiselessTrivial, result.a.coefsT)
    comp(result.a.coefsT, result.current_variances, mu.coefsT)


# TODO: can be made faster by using local memory
class TLweExtractLweSample(Computation):

    def __init__(self, tlwe_a):

        assert len(tlwe_a.shape) > 2
        k = tlwe_a.shape[-2] - 1
        N = tlwe_a.shape[-1]

        batch_shape = tlwe_a.shape[:-2]
        a = Type(tlwe_a.dtype, batch_shape + (k * N,))
        b = Type(tlwe_a.dtype, batch_shape)

        self._fill_a = PureParallel(
            [Parameter('a', Annotation(a, 'o')),
            Parameter('tlwe_a', Annotation(tlwe_a, 'i'))],
            """
            ${a.ctype} a;
            if (${idxs[-1]} == 0)
            {
                a = ${tlwe_a.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-2]}, 0);
            }
            else
            {
                a = -${tlwe_a.load_idx}(${", ".join(idxs[:-2])}, ${idxs[-2]}, ${N} - ${idxs[-1]});
            }
            ${a.store_idx}(${", ".join(idxs[:-2])}, ${idxs[-2]} * ${N} + ${idxs[-1]}, a);
            """,
            render_kwds=dict(k=k, N=N),
            guiding_array=batch_shape + (k, N))

        self._fill_b = PureParallel(
            [Parameter('b', Annotation(b, 'o')),
            Parameter('tlwe_a', Annotation(tlwe_a, 'i'))],
            """
            ${b.store_same}(${tlwe_a.load_idx}(${", ".join(idxs)}, ${k}, 0));
            """,
            render_kwds=dict(k=k))

        Computation.__init__(self,
            [Parameter('a', Annotation(a, 'o')),
            Parameter('b', Annotation(b, 'o')),
            Parameter('tlwe_a', Annotation(tlwe_a, 'i'))])

    def _build_plan(self, plan_factory, device_params, a, b, tlwe_a):
        plan = plan_factory()
        plan.computation_call(self._fill_a, a, tlwe_a)
        plan.computation_call(self._fill_b, b, tlwe_a)
        return plan


def tLweExtractLweSample_gpu(
        result: LweSampleArray, x: TLweSampleArray, params: LweParams, rparams: TLweParams):
    thr = result.a.thread
    comp = get_computation(thr, TLweExtractLweSample, x.a.coefsT)
    comp(result.a, result.b, x.a.coefsT)


def tLweMulByXaiMinusOne_gpu(result:TLweSampleArray, ai, ai_idx, bk: TLweSampleArray, params: TLweParams):
    # TYPING: ai::Array{Int32}
    tp_mul_by_xai_minus_one_gpu(result.a, ai, ai_idx, bk.a)


# result = result + sample
def tLweAddTo_gpu(result: TLweSampleArray, accum: TLweSampleArray, params: TLweParams):
    result.a.coefsT += accum.a.coefsT
    result.current_variances += accum.current_variances


# result = sample
def tLweCopy_gpu(result: TLweSampleArray, sample: TLweSampleArray, params: TLweParams):
    thr = result.a.coefsT.thread
    thr.copy(sample.a.coefsT, dest=result.a.coefsT)
    thr.copy(sample.current_variances, dest=result.current_variances)
