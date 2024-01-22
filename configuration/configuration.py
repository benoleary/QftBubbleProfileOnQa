from typing import List, Optional
from dataclasses import dataclass
from hamiltonian.field import FieldDefinition


class QftModelConfiguration:
    def __init__(
            self,
            *,
            first_field: FieldDefinition,
            second_field: Optional[FieldDefinition] = None,
            potential_in_quartic_GeV_per_field_step: List[List[float]]
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
        self.potential_in_quartic_GeV_per_field_step = (
            potential_in_quartic_GeV_per_field_step
        )


@dataclass(kw_only=True, frozen=True, repr=False, eq=False)
class SpatialLatticeConfiguration:
    number_of_spatial_steps: int
    spatial_step_in_inverse_GeV: float
    volume_exponent: int


@dataclass(kw_only=True, frozen=True, repr=False, eq=False)
class AnnealerConfiguration:
    sampler_name: str = "default"
    number_of_shots: Optional[int] = None
    output_CSV_filename: Optional[str] = None
    command_for_gnuplot: Optional[str] = None
