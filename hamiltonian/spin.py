from typing import List, Optional
from hamiltonian.hamiltonian import AnnealerHamiltonian
from hamiltonian.field import FieldDefinition
from minimization.weight import WeightAccumulator


class SpinHamiltonian(AnnealerHamiltonian):
    def __init__(
            self,
            *,
            first_field: FieldDefinition,
            second_field: Optional[FieldDefinition] = None
    ):




    def weights_for_domain_wall(
            self,
            *,
            number_of_spins: int,
            end_spin_weight: float,
            spin_alignment_weight: float
    ) -> WeightAccumulator:
        """
        This returns the weights to ensure that the spins are valid for the
        Ising-chain domain wall model, in a form that can be combined with a
        FieldAtPoint to create a pair of dicts in the form for sample_ising: a
        dict of linear biases, which could be represented by a vector, and a
        dict of quadratic biases, which could be represented as an
        upper-triangular matrix of correlation weights, with zeros on the
        diagonal. (Apparently it is not necessary that the dict is
        "upper-triangular"; the middleware seems to cope.)
        """
        # This is the case of a FieldAtPoint having correlations between its
        # own variables.
        spin_weights = WeightAccumulator(
            first_number_of_values=number_of_spins,
            second_number_of_values=number_of_spins
        )

        # First, we set the weights to fix the ends so that there is a domain of
        # 1s from the first index and a domain of 0s ending at the last index.
        # The signs are this way because we want the first spin to be |1> which
        # multiplies its weight by -1 in the objective function, and the last
        # spin to be |0> which multiplies its weight by +1.
        spin_weights.first_linear_weights[0] = end_spin_weight
        spin_weights.first_linear_weights[-1] = -end_spin_weight

        # Next, each pair of nearest neighbors gets weighted to favor having the
        # same values - which is either (-1)^2 or (+1)^2, so +1, while opposite
        # values multiply the weighting by (-1) * (+1) = -1. Therefore, a
        # negative weighting will penalize opposite spins with a positive
        # contribution to the objective function.
        for i in range(number_of_spins - 1):
            spin_weights.quadratic_weights[i][i + 1] = -spin_alignment_weight

        return spin_weights

    def weights_for_single_field_potential_at_point(
            potential_in_quartic_GeV_per_field_step: List[float]
    ) -> WeightAccumulator:
        """
        This function adds calculates weights for spin variables of a single
        field represented by a set of FieldAtPoint objects, one for every
        spatial point. (The calculation is significantly different when dealing
        with multiple fields at a single spatial point.)
        """
        # If we have 4 values U_0, U_1, U_2, and U_3, the field has 5 binary
        # variables, where the first and last are fixed, and there are 4 valid
        # bitstrings: 10000, 11000, 11100, and 11110. If the middle 3 spin
        # variables have weights A, B, and C respectively, then we have the
        # following simultaneous equations when considering the weights brought
        # to the objective function:
        # +A+B+C = U_0
        # -A+B+C = U_1
        # -A-B+C = U_2
        # -A-B-C = U_3
        # Therefore,
        # U_3 - U_2 = -2 C
        # U_2 - U_1 = -2 B
        # U_1 - U_0 = -2 A
        # and so we end up with the factor 0.5 and the potential for the lower
        # index minus the potential for the higher index.
        previous_value = potential_in_quartic_GeV_per_field_step[0]
        next_values = potential_in_quartic_GeV_per_field_step[1:]

        # This is the case of a single FieldAtPoint with no quadratic weights.
        spin_weights = WeightAccumulator(
            first_number_of_values=len(next_values),
            second_number_of_values=0
        )

        for i, next_value in enumerate(next_values):
            spin_weights[i] = 0.5 * (previous_value - next_value)

        return spin_weights

    def weights_for_kinetic_term(
            *,
            number_of_spins: int,
            field_step_in_GeV: float,
            radius_step_in_inverse_GeV: float
    ) -> WeightAccumulator:
        """
        This gives the weights for the part of the kinetic term for a single
        field over a single step in radius. We could try to arrange things so
        that each radius has a value for its field(s) and a value for the field
        difference(s) but in the end it would amount to the same thing while
        making intermediate steps more complicated.
        However, we bother only with correlations between spins which can change
        under the ICDW model, so we give no weight to correlations of the first
        or last spins of each field.
        """
        # We assume that the field has the same step size at both spatial
        # points.
        absolute_weight = (
            (0.125 * field_step_in_GeV * field_step_in_GeV)
            / (radius_step_in_inverse_GeV * radius_step_in_inverse_GeV)
        )

        for left in (at_smaller_radius, at_larger_radius):
            for right in (at_smaller_radius, at_larger_radius):
                signed_weight = (
                    absolute_weight if (left == right) else -absolute_weight
                )
                for l in left.binary_variable_names[1:-1]:
                    for r in right.binary_variable_names[1:-1]:
                        weights[(l, r)] = signed_weight
        return BiasAccumulator(initial_quadratics=weights)
