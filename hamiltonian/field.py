import minimization.variable
from minimization.weight import BiasAccumulator

class FieldAtPoint:
    """
    This class represents the strength of a QFT scalar field at a point in
    space-time using spin variables in the Ising chain domain wall model. The
    field takes (number_of_values_for_field + 1) spin variables so that there
    can be a single domain wall, with down spins (normally represented as |1> or
    sometimes informally as 1) on the lower-index side and up spins (|0> or 0)
    on the higher-index side.
    """
    def __init__(
            self,
            *,
            field_name: str,
            spatial_point_identifier: str,
            number_of_values_for_field: int,
            field_step_in_GeV: float
        ):
        """
        The constructor just sets up the names for the spin variables, since the
        sample_ising methods of D-Wave samplers just want dicts of names of spin
        variables mapped to linear biases and dicts of pairs of names of spin
        variables mapped to quadratic biases.
        """
        if number_of_values_for_field < 2:
            raise ValueError("Need a range of at least 2 values for the field")
        self.field_name = field_name
        self.spatial_point_identifier = spatial_point_identifier
        # The variable names are indexed from zero, so if we have say 10 values
        # for the field, we actually index 0 to 9 so use only 1 digit.
        name_function = minimization.variable.name_for_index(
            f"{field_name}_{spatial_point_identifier}_",
            number_of_values_for_field - 1
        )
        # We need a binary variable fixed to |1> at the start and another fixed
        # to |0> at the end, in addition to the variables which can actually
        # vary.
        self.binary_variable_names = [
            name_function(i) for i in range(number_of_values_for_field + 1)
        ]
        self.field_step_in_GeV = field_step_in_GeV

    def weights_for_domain_wall(
            self,
            *,
            end_spin_weight: float,
            spin_alignment_weight: float
        ) -> BiasAccumulator:
        """
        This returns the weights to ensure that the spins are valid for the
        Ising-chain domain wall model, in the form for sample_ising: a dict of
        linear biases, which could be represented by a vector, and a dict of
        quadratic biases, which could be represented as an upper-triangular
        matrix of correlation weights, with zeros on the diagonal.
        """
        # First, we set the weights to fix the ends so that there is a domain of
        # 1s from the first index and a domain of 0s ending at the last index.
        # The signs are this way because we want the first spin to be |1> which
        # multiplies its weight by -1 in the objective function, and the last
        # spin to be |0> which multiplies its weight by +1.
        first_variable = self.binary_variable_names[0]
        last_variable = self.binary_variable_names[-1]
        spin_biases = BiasAccumulator(
            initial_linears={
                first_variable: end_spin_weight,
                last_variable: -end_spin_weight
            }
        )
        # Next, each pair of nearest neighbors gets weighted to favor having the
        # same values - which is either (-1)^2 or (+1)^2, so +1, while opposite
        # values multiply the weighting by (-1) * (+1) = -1. Therefore, a
        # negative weighting will penalize opposite spins with a positive
        # contribution to the objective function.
        lower_variable = first_variable
        for higher_variable in self.binary_variable_names[1:]:
            spin_biases.add_quadratics({
                (lower_variable, higher_variable): -spin_alignment_weight
            })
            lower_variable = higher_variable
        return spin_biases

    def weights_for_fixed_value(
            self,
            *,
            fixing_weight: float,
            number_of_down_spins: int
        ) -> BiasAccumulator:
        """
        This returns the weights to fix the spins so that there are
        number_of_down_spins |1>s. Negative numbers can be given to instead
        specify -number_of_down_spins |0>s, similar to negative indices in a
        Python array.
        """
        spin_biases = BiasAccumulator(
            initial_linears={
                binary_variable_name: fixing_weight
                for binary_variable_name
                in self.binary_variable_names[:number_of_down_spins]
            }
        )
        spin_biases.add_linears({
                binary_variable_name: -fixing_weight
                for binary_variable_name
                in self.binary_variable_names[number_of_down_spins:]
            }
        )
        return spin_biases
