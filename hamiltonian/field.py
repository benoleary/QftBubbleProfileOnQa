from typing import Dict
import minimization.variable
from minimization.weight import BiasAccumulator


class FieldDefinition:
    def __init__(
            self,
            *,
            field_name: str,
            number_of_values: int,
            lower_bound_in_GeV: float,
            upper_bound_in_GeV: float,
            true_vacuum_value_in_GeV: float,
            false_vacuum_value_in_GeV: float
        ):
        if number_of_values < 2:
            raise ValueError("Need a range of at least 2 values for the field")
        if lower_bound_in_GeV >= upper_bound_in_GeV:
            raise ValueError(
                "Need lower bound for field to be below upper bound"
            )

        def in_bounds(value_in_GeV: float) -> bool:
            return (lower_bound_in_GeV <= value_in_GeV <= upper_bound_in_GeV)

        true_vacuum_in_bounds = in_bounds(true_vacuum_value_in_GeV)
        false_vacuum_in_bounds = in_bounds(false_vacuum_value_in_GeV)
        if not (true_vacuum_in_bounds and false_vacuum_in_bounds):
            raise ValueError("Neither vacuum can be outside bounds for field")
        self.field_name = field_name
        self.number_of_values = number_of_values
        self.lower_bound_in_GeV = lower_bound_in_GeV
        self.upper_bound_in_GeV = upper_bound_in_GeV
        self.true_vacuum_value_in_GeV = true_vacuum_value_in_GeV
        self.false_vacuum_value_in_GeV = false_vacuum_value_in_GeV
        # The field step size is positive because
        # upper_bound_in_GeV > lower_bound_in_GeV and
        # number_of_values >= 2.
        self.step_in_GeV = (
            (upper_bound_in_GeV - lower_bound_in_GeV) / (number_of_values - 1)
        )
        self.true_vacuum_value_in_steps = self._get_vacuum_value_in_steps(
            true_vacuum_value_in_GeV
        )
        self.false_vacuum_value_in_steps = self._get_vacuum_value_in_steps(
            false_vacuum_value_in_GeV
        )

    def _get_vacuum_value_in_steps(self, vacuum_value_in_GeV: float) -> int:
        # The vacuum in steps is positive because
        # vacuum_value_in_GeV >= lower_bound_in_GeV and
        # the field step size is positive.
        vacuum_value_in_steps = int(
            (vacuum_value_in_GeV - self.lower_bound_in_GeV)
            / self.step_in_GeV
        )
        # We check whether rounding up instead of down would get closer to the
        # value in GeV.
        reconstructed_value_in_GeV = (
            (vacuum_value_in_steps * self.step_in_GeV) + self.lower_bound_in_GeV
        )
        reconstructed_difference_in_GeV = (
            vacuum_value_in_GeV - reconstructed_value_in_GeV
        )
        return (
            vacuum_value_in_steps
            if reconstructed_difference_in_GeV <= (0.5 * self.step_in_GeV)
            else (vacuum_value_in_steps + 1)
        )


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
            field_definition: FieldDefinition,
            spatial_point_identifier: str
        ):
        """
        The constructor just sets up the names for the spin variables, since the
        sample_ising methods of D-Wave samplers just want dicts of names of spin
        variables mapped to linear biases and dicts of pairs of names of spin
        variables mapped to quadratic biases.
        """
        self.field_definition = field_definition
        # The variable names are indexed from zero, so if we have say 10 values
        # for the field, we actually index 0 to 9 so use only 1 digit.
        name_function = minimization.variable.name_for_index(
            f"{field_definition.field_name}_{spatial_point_identifier}_",
            field_definition.number_of_values - 1
        )
        # We need a binary variable fixed to |1> at the start and another fixed
        # to |0> at the end, in addition to the variables which can actually
        # vary.
        self.binary_variable_names = [
            name_function(i)
            for i in range(field_definition.number_of_values + 1)
        ]

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
        matrix of correlation weights, with zeros on the diagonal. (Apparently
        it is not necessary that the dict is "upper-triangular", the middleware
        seems to cope.)
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
        if number_of_down_spins == 0:
            raise ValueError(
                "Input of 0 (or -0) should set all spins to |1> (or |0>) but"
                " this would prevent a domain wall"
            )
        if number_of_down_spins > self.field_definition.number_of_values:
            raise ValueError(
                f"At most {self.field_definition.number_of_values} can be set"
                f" to |1>, {number_of_down_spins} were requested"
            )
        if -number_of_down_spins > self.field_definition.number_of_values:
            raise ValueError(
                f"At most {self.field_definition.number_of_values} can be set"
                f" to |0>, {-number_of_down_spins} were requested (as negative"
                " input)"
            )
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
        })
        return spin_biases

    def in_GeV(self, spins_from_sample: Dict[str, int]):
        # This adds field_step_in_GeV to offset_from_origin_in_GeV for every |1>
        # beyond the fixed first one (and we ignore the fixed last |0>).
        total_in_GeV = self.field_definition.lower_bound_in_GeV
        for variable_name in self.binary_variable_names[1:-1]:
            if spins_from_sample.get(variable_name, 0) < 0:
                total_in_GeV += self.field_definition.step_in_GeV
        return total_in_GeV
