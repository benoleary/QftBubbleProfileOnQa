from typing import List, Optional


class DiscreteConfiguration:
    def __init__(
            self,
            *,
            first_field_name: str,
            second_field_name: Optional[str] = None,
            number_of_spatial_steps: int,
            spatial_step_in_inverse_GeV: float,
            field_step_in_GeV: float,
            potential_in_quartic_GeV_per_field_step: List[float]
        ):
        if not potential_in_quartic_GeV_per_field_step:
            raise ValueError("Cannot have a potential without any values")
        self.first_field_name = first_field_name
        self.second_field_name = second_field_name
        self.number_of_spatial_steps = number_of_spatial_steps
        self.spatial_step_in_inverse_GeV = spatial_step_in_inverse_GeV
        self.field_step_in_GeV = field_step_in_GeV
        self.potential_in_quartic_GeV_per_field_step = (
            potential_in_quartic_GeV_per_field_step
        )
        self.number_of_values_for_field = len(
            potential_in_quartic_GeV_per_field_step
        )
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
