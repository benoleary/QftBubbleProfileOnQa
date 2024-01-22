from hamiltonian.field import FieldAtPoint
from minimization.weight import WeightAccumulator

class SpinDomainWallWeighter:
    """
    This class implements the methods of the DomainWallWeighter Protocol,
    providing weights which should be used as input for the sample_ising method
    of a dimod.Sampler, as spin variables are assumed.
    """
    def weights_for_domain_wall(
            self,
            *,
            field_at_point: FieldAtPoint,
            end_spin_weight: float,
            spin_alignment_weight: float
    ) -> WeightAccumulator:
        """
        This returns the weights to ensure that the spins of the given
        FieldAtPoint are valid for the Ising-chain domain wall model, in the
        form for sample_ising: a dict of linear biases, which could be
        represented by a vector, and a dict of quadratic biases, which could be
        represented as an upper-triangular matrix of correlation weights, with
        zeros on the diagonal. (Apparently it is not necessary that the dict is
        "upper-triangular", the middleware seems to cope.)
        """
        # First, we set the weights to fix the ends so that there is a domain of
        # 1s from the first index and a domain of 0s ending at the last index.
        # The signs are this way because we want the first spin to be |1> which
        # multiplies its weight by -1 in the objective function, and the last
        # spin to be |0> which multiplies its weight by +1.
        first_variable = field_at_point.binary_variable_names[0]
        last_variable = field_at_point.binary_variable_names[-1]
        spin_biases = WeightAccumulator(
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
        for higher_variable in field_at_point.binary_variable_names[1:]:
            spin_biases.add_quadratics({
                (lower_variable, higher_variable): -spin_alignment_weight
            })
            lower_variable = higher_variable
        return spin_biases

    def weights_for_fixed_value(
            self,
            *,
            field_at_point: FieldAtPoint,
            fixing_weight: float,
            number_of_ones: int
    ) -> WeightAccumulator:
        """
        This returns the weights to fix the spins of the given FieldAtPoint so
        that there are number_of_ones |1>s. Negative numbers can be given to
        instead specify -number_of_ones |0>s, similar to negative indices in a
        Python array.
        """
        if number_of_ones == 0:
            raise ValueError(
                "Input of 0 (or -0) should set all spins to |1> (or |0>) but"
                " this would prevent a domain wall"
            )
        number_of_values = field_at_point.field_definition.number_of_values
        if number_of_ones > number_of_values:
            raise ValueError(
                f"At most {number_of_values} can be set to |1>,"
                f" {number_of_ones} were requested"
            )
        if -number_of_ones > number_of_values:
            raise ValueError(
                f"At most {number_of_values} can be set to |0>,"
                f" {-number_of_ones} were requested (as negative input)"
            )
        spin_biases = WeightAccumulator(
            initial_linears={
                binary_variable_name: fixing_weight
                for binary_variable_name
                in field_at_point.binary_variable_names[:number_of_ones]
            }
        )
        spin_biases.add_linears({
            binary_variable_name: -fixing_weight
            for binary_variable_name
            in field_at_point.binary_variable_names[number_of_ones:]
        })
        return spin_biases
