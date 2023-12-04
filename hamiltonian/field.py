from typing import List, Tuple
from minimization.weight import AccumulatingWeightMatrix

class FieldAtPoint:
    """
    This class represents the strength of a QFT scalar field at a point in
    space-time using binary variables in the Ising chain domain wall model.
    The field takes up two extra variables beyond the given
    number_of_active_binary_variables so that there can be a single domain wall.
    """
    def __init__(
            self,
            *,
            field_name: str,
            spatial_point_identifier: str,
            number_of_active_binary_variables: int
        ):
        """
        The constructor just sets up the names for the binary variables, since
        the D-Wave samplers just want dicts of names of binary variables or
        pairs of names of binary variables, mapped to weights.
        """
        if number_of_active_binary_variables < 1:
            raise ValueError("Number of active binary variables must be > 0")
        self.field_name = field_name
        self.spatial_point_identifier = spatial_point_identifier
        self.number_of_active_binary_variables = \
            number_of_active_binary_variables
        # We need a binary variable fixed to 1 at the start and another fixed to
        # 0 at the end, in addition to the variables which can actually vary.
        self.binary_variable_names = [
            f"{field_name}_{spatial_point_identifier}_{i}"
            for i in range(number_of_active_binary_variables + 2)
        ]
        normalization_factor = 1.0/number_of_active_binary_variables
        # We do not bother with a value for the last binary variable, which is
        # fixed at 0.
        self.binary_variable_names_with_field_values = (
            [(self.binary_variable_names[0], 0.0)]
            + [
                (self.binary_variable_names[i], i * normalization_factor)
                for i in range(1, number_of_active_binary_variables)
            ]
            + [(self.binary_variable_names[-2], 1.0)]
        )

    def weights_for_ICDW(
            self,
            *,
            end_spin_weight: float,
            spin_alignment_weight: float
        ) -> AccumulatingWeightMatrix:
        """
        This returns the matrix of weights in the form expected by D-Wave Ocean
        stuff: an upper-triangular matrix of correlation weights, where the
        diagonal elements are understood to be actually the weights of the
        linear terms.
        """
        # We could do this with fewer assignments, but it won't be the
        # bottleneck while it is important to make sure that it is correct.
        # First, we set the weights to fix the ends so that there is a domain of
        # 1s from the first index and a domain of 0s ending at the last index.
        first_variable = self.binary_variable_names[0]
        last_variable = self.binary_variable_names[-1]
        weights_matrix = AccumulatingWeightMatrix({
            (first_variable, first_variable): -end_spin_weight,
            (last_variable, last_variable): end_spin_weight
        })
        # Next, each pair of nearest neighbors gets weighted to favor same
        # values, by penalizing each individual but with a cancelling weight for
        # when both are 1. (There is nothing to cancel if both are 0.)
        lower_variable = first_variable
        for higher_variable in self.binary_variable_names[1:]:
            weights_matrix.add({
                (lower_variable, lower_variable): spin_alignment_weight,
                (lower_variable, higher_variable): -2.0 * spin_alignment_weight,
                (higher_variable, higher_variable): spin_alignment_weight
            })
            lower_variable = higher_variable
        return weights_matrix
