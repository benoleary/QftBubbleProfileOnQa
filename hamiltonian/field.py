from typing import Dict, Tuple

class Field:
    """
    This class represents the strength of a QFT scalar field at a point in
    space-time.
    """
    def __init__(
            self,
            *,
            field_name: str,
            spatial_point_identifier: str,
            number_of_binary_variables: input
        ):
        """
        The constructor just sets up the names for the binary variables, since
        the D-Wave samplers just want dicts of names of binary variables or
        pairs of names of binary variables, mapped to weights.
        """
        self.field_name = field_name
        self.spatial_point_identifier = spatial_point_identifier
        self.number_of_binary_variables = number_of_binary_variables
        self.binary_variable_names = [
            f"{field_name}_{spatial_point_identifier}_{binary_variable_index}"
            for binary_variable_index in range(number_of_binary_variables)
        ]

    def get_weights_for_ICDW(
            self,
            *,
            end_spin_weight: float,
            spin_alignment_weight: float
        ) -> Dict[Tuple[str, str], float]:
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
        weights_matrix = {
            (first_variable, first_variable): -end_spin_weight,
            (last_variable, last_variable): end_spin_weight
        }
        # Next, each pair of nearest neighbors gets weighted to favor same
        # values, by penalizing each individual but with a cancelling weight for
        # when both are 1. (There is nothing to cancel if both are 0.)
        for index_between in range(self.number_of_binary_variables - 1):
            lower_variable = self.binary_variable_names[index_between]
            higher_variable = self.binary_variable_names[index_between + 1]
            for individual_variable in (lower_variable, higher_variable):
                diagonal_tuple = (individual_variable, individual_variable)
                weights_matrix[diagonal_tuple] = (
                    weights_matrix.get(diagonal_tuple, 0.0)
                    + spin_alignment_weight
                )
            weights_matrix[(lower_variable, higher_variable)] = \
                -2.0 * spin_alignment_weight
        return weights_matrix
