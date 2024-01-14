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
    However, we bother only with correlations between spins which can change
    under the ICDW model, so we give no weight to correlations of the first or
    last spins of each field.
    """
    weights = {}
    # We assume that both fields have the same step sizes.
    absolute_weight = (
        (
            0.125
            * at_smaller_radius.field_definition.step_in_GeV
            * at_larger_radius.field_definition.step_in_GeV
        )
        / (radius_difference_in_inverse_GeV * radius_difference_in_inverse_GeV)
    )

    for left in (at_smaller_radius, at_larger_radius):
        for right in (at_smaller_radius, at_larger_radius):
            signed_weight = (
                absolute_weight if left == right else -absolute_weight
            )
            for l in left.binary_variable_names[1:-1]:
                for r in right.binary_variable_names[1:-1]:
                    weights[(l, r)] = signed_weight
    return BiasAccumulator(initial_quadratics=weights)
