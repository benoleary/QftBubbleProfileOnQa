from typing import List
from minimization.weight import BiasAccumulator
from hamiltonian.field import FieldAtPoint

def weights_for(
        *,
        potential_in_quartic_GeV_per_field_step: List[float],
        single_field: FieldAtPoint
    ) -> BiasAccumulator:
        """
        This function adds calculates weights for spin variables of a single
        field represented by a set of FieldAtPoint objects, one for every
        spatial point. (The calculation is significantly different when dealing
        with multiple fields at a single spatial point.)
        """
        number_of_potential_values = len(
            potential_in_quartic_GeV_per_field_step
        )
        # If there are N binary variables for the field, at least 1 and at most
        # (N - 1) of them will be 1 and the rest 0, so there are (N - 1) values
        # possible for the field, which must match the number of values provided
        # for the potential.
        number_of_field_values = len(single_field.binary_variable_names) - 1
        if number_of_potential_values != number_of_field_values:
            raise ValueError(
                f"Provided {number_of_potential_values} values for potential"
                f" but {number_of_field_values} values for field"
            )
        linear_weights = {}
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
        for i in range(1, number_of_field_values):
            linear_weights[single_field.binary_variable_names[i]] = (
                0.5 * (
                    potential_in_quartic_GeV_per_field_step[i - 1]
                    - potential_in_quartic_GeV_per_field_step[i]
                )
            )
        return BiasAccumulator(initial_linears=linear_weights)
