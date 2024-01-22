from typing import Dict
import minimization.variable
from minimization.weight import WeightAccumulator


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
    space-time in the Ising chain domain wall model. The field is represented by
    (number_of_values_for_field + 1) variables so that there can be a single
    domain wall, with down spins or on bits (normally represented as |1> or
    sometimes informally as 1) on the lower-index side and up spins or off bits
    (|0> or 0) on the higher-index side.
    """
    def __init__(
            self,
            *,
            field_definition: FieldDefinition,
            spatial_point_identifier: str
    ):
        """
        The constructor just sets up the names for the variables, since the
        sampling methods of D-Wave samplers just want dicts of pairs of names
        of variables mapped to quadratic biases and, depending on whether using
        spin or bit variables, dicts of spin variables mapped to linear weights
        or more entries in the dict of pairs of names but with bit variables
        self-correlating mapped to their linear biases.
        """
        self.field_definition = field_definition
        # The variable names are indexed from zero, so if we have say 10 values
        # for the field, we actually index 0 to 9 so use only 1 digit.
        name_function = minimization.variable.name_for_index(
            f"{field_definition.field_name}_{spatial_point_identifier}_",
            field_definition.number_of_values - 1
        )
        # We need a variable fixed to |1> at the start and another fixed to |0>
        # at the end, in addition to the variables which can actually vary.
        self.binary_variable_names = [
            name_function(i)
            for i in range(field_definition.number_of_values + 1)
        ]

    def in_GeV(self, spins_from_sample: Dict[str, int]):
        # This adds field_step_in_GeV to offset_from_origin_in_GeV for every |1>
        # beyond the fixed first one (and we ignore the fixed last |0>).
        total_in_GeV = self.field_definition.lower_bound_in_GeV
        for variable_name in self.binary_variable_names[1:-1]:
            if spins_from_sample.get(variable_name, 0) < 0:
                total_in_GeV += self.field_definition.step_in_GeV
        return total_in_GeV
