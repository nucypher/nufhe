import numpy

from reikna.algorithms import PureParallel
from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.cluda import dtypes

from .computation_cache import get_computation


# Declaring the types here instead of `numeric_functions.py` to avoid a cricular import.

# Element on a torus
Torus32 = numpy.int32

# Accompanying integer type (same size)
Int32 = numpy.int32

# The type for floating-point values (e.g., errors)
Float = numpy.float64


class Torus32ToPhase(Computation):

    def __init__(self, shape, mspace_size):

        self._mspace_size = mspace_size

        messages = Type(Torus32, shape)
        result = Type(Int32, shape)

        Computation.__init__(self, [
            Parameter('result', Annotation(result, 'o')),
            Parameter('messages', Annotation(messages, 'i'))])

    def _build_plan(self, plan_factory, device_params, result, phase):
        plan = plan_factory()

        tr = Transformation(
            [
                Parameter('result', Annotation(result, 'o')),
                Parameter('phase', Annotation(phase, 'i')),
            ],
            """
            <%
                interv = 2**32 // mspace_size
                half_interv = interv // 2
            %>
            ${phase.ctype} phase = ${phase.load_same};
            ${result.store_same}(((unsigned int)phase + ${half_interv}) / ${interv});
            """,
            render_kwds=dict(
                mspace_size=self._mspace_size,
                uint64=dtypes.ctype(numpy.uint64)),
            connectors=['result', 'phase'])

        plan.computation_call(
            PureParallel.from_trf(tr, guiding_array='result'),
            result, phase)

        return plan
