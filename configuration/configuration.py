from typing import List, Optional
from hamiltonian.field import FieldDefinition


class DiscreteConfiguration:
    def __init__(
            self,
            *,
            first_field: FieldDefinition,
            second_field: Optional[FieldDefinition] = None,
            number_of_spatial_steps: int,
            spatial_step_in_inverse_GeV: float,
            volume_exponent: int,
            potential_in_quartic_GeV_per_field_step: List[List[float]],
            sampler_name: Optional[str] = None,
            number_of_shots: Optional[int] = None,
            output_CSV_filename: Optional[str] = None,
            command_for_gnuplot: Optional[str] = None
        ):
        self.number_of_values_for_second_field = len(
            potential_in_quartic_GeV_per_field_step
        )
        self.number_of_values_for_first_field = (
            0 if not self.number_of_values_for_second_field
            else len(potential_in_quartic_GeV_per_field_step[0])
        )
        if not self.number_of_values_for_first_field:
            raise ValueError("Cannot have a potential without any values")
        if second_field and self.number_of_values_for_second_field < 2:
            raise ValueError(
                "Second field defined but only one row of potential values"
                " (i.e. there is only one value allowed for the second field)"
            )
        self.first_field = first_field
        self.second_field = second_field
        self.number_of_spatial_steps = number_of_spatial_steps
        self.spatial_step_in_inverse_GeV = spatial_step_in_inverse_GeV
        self.volume_exponent = volume_exponent
        self.potential_in_quartic_GeV_per_field_step = (
            potential_in_quartic_GeV_per_field_step
        )
        self.minimum_potential = potential_in_quartic_GeV_per_field_step[0][0]
        self.maximum_potential = self.minimum_potential
        for potential_row in self.potential_in_quartic_GeV_per_field_step:
            for potential_value in potential_row:
                if potential_value < self.minimum_potential:
                    self.minimum_potential = potential_value
                if potential_value > self.maximum_potential:
                    self.maximum_potential = potential_value
        self.maximum_potential_weight_difference = (
            self.maximum_potential - self.minimum_potential
        )
        self.sampler_name = sampler_name
        self.number_of_shots = number_of_shots
        self.output_CSV_filename = output_CSV_filename
        self.command_for_gnuplot = command_for_gnuplot
