from typing import List, Optional


class DiscreteConfiguration:
    def __init__(
            self,
            *,
            first_field_name: str,
            first_field_step_in_GeV: float,
            first_field_offset_in_GeV: float,
            second_field_name: Optional[str] = None,
            second_field_step_in_GeV: Optional[float] = None,
            second_field_offset_in_GeV: Optional[float] = None,
            number_of_spatial_steps: int,
            spatial_step_in_inverse_GeV: float,
            volume_exponent: int,
            potential_in_quartic_GeV_per_field_step: List[float],
            sampler_name: Optional[str] = None,
            number_of_shots: Optional[int] = None,
            output_CSV_filename: Optional[str] = None,
            command_for_gnuplot: Optional[str] = None
        ):
        if not potential_in_quartic_GeV_per_field_step:
            raise ValueError("Cannot have a potential without any values")
        self.first_field_name = first_field_name
        self.first_field_step_in_GeV = first_field_step_in_GeV
        self.first_field_offset_in_GeV = first_field_offset_in_GeV
        self.second_field_name = second_field_name
        self.second_field_step_in_GeV = second_field_step_in_GeV
        self.second_field_offset_in_GeV = second_field_offset_in_GeV
        self.number_of_spatial_steps = number_of_spatial_steps
        self.spatial_step_in_inverse_GeV = spatial_step_in_inverse_GeV
        self.volume_exponent = volume_exponent

        # TODO: account for second field in all the following, down to
        # self.maximum_weight_difference
        self.potential_in_quartic_GeV_per_field_step = (
            potential_in_quartic_GeV_per_field_step
        )
        self.number_of_values_for_first_field = len(
            potential_in_quartic_GeV_per_field_step
        )
        self.number_of_values_for_second_field = 0
        self.minimum_potential = potential_in_quartic_GeV_per_field_step[0]
        self.maximum_potential = potential_in_quartic_GeV_per_field_step[0]
        for potential_value in potential_in_quartic_GeV_per_field_step:
            if potential_value < self.minimum_potential:
                self.minimum_potential = potential_value
            if potential_value > self.maximum_potential:
                self.maximum_potential = potential_value
        self.maximum_weight_difference = (
            self.maximum_potential - self.minimum_potential
        )

        self.sampler_name = sampler_name
        self.number_of_shots = number_of_shots
        self.output_CSV_filename = output_CSV_filename
        self.command_for_gnuplot = command_for_gnuplot
