import numpy

from reikna import helpers
from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.algorithms import PureParallel
from reikna.cluda import dtypes

from .numeric_functions import Torus32, Int32


TEMPLATE = helpers.template_for(__file__)


class ShiftTorusPolynomial(Computation):
    """
    Calculate batched

        `result = (X^((2*N - p) if invert_powers else p) - minus_one) * source`

    where `N` is the polynomial degree.

    If `powers_view` is `True`, a view of `powers` is taken
    using fixed `powers_idx` as the innermost index.
    """

    def __init__(
            self, polynomial_degree, shape, powers_shape,
            powers_view=False, minus_one=False, invert_powers=False):

        self._batch_shape = powers_shape[:-1] if powers_view else powers_shape
        assert self._batch_shape == shape[:len(self._batch_shape)]

        self._powers_view = powers_view
        self._minus_one = minus_one
        self._invert_powers = invert_powers

        polynomials = Type(Torus32, shape + (polynomial_degree,))
        powers = Type(Int32, powers_shape)

        Computation.__init__(self, [
            Parameter('result', Annotation(polynomials, 'o')),
            Parameter('source', Annotation(polynomials, 'i')),
            Parameter('powers', Annotation(powers, 'i')),
            Parameter('powers_idx', Annotation(Type(Int32))) # unused if powers_view==False
            ])

    def _build_plan(self, plan_factory, device_params, result, source, powers, powers_idx):

        poly_batch_shape = result.shape[len(self._batch_shape):-1]
        polynomial_degree = result.shape[-1]

        plan = plan_factory()
        plan.kernel_call(
            TEMPLATE.get_def("shift_torus_polynomial"),
            [result, source, powers, powers_idx],
            global_size=(
                helpers.product(self._batch_shape),
                helpers.product(poly_batch_shape),
                polynomial_degree),
            render_kwds=dict(
                batch_len=len(self._batch_shape),
                poly_batch_len=len(poly_batch_shape),
                polynomial_degree=polynomial_degree,
                powers_view=self._powers_view,
                minus_one=self._minus_one,
                invert_powers=self._invert_powers))

        return plan
