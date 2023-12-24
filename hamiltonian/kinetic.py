from typing import List
from minimization.weight import BiasAccumulator
from hamiltonian.field import FieldAtPoint

def weights_for_difference(
        *,
        at_smaller_radius: FieldAtPoint,
        at_larger_radius: FieldAtPoint,
        radius_difference_in_inverse_GeV: float
    ) -> BiasAccumulator:
    """
    This gives the weights for the part of the kinetic term for a single field
    over a single step in radius. We could try to arrange things so that each
    radius has a value for its field(s) and a value for the field difference(s)
    but in the end it would amount to the same thing while making intermediate
    steps more complicated.
    """
    weights = {}
    return BiasAccumulator(initial_quadratics=weights)
