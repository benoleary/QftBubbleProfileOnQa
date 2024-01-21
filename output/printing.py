from typing import Dict, List
from dimod import SampleSet

from structure.bubble import BubbleProfile
from basis.field import FieldCollectionAtPoint


class CsvWriter:
    """
    This class writes a given solution for a bubble profile into a CSV file.
    """
    def __init__(
            self,
            *,
            bubble_profile: BubbleProfile,
            separation_character: str = ";"
    ):
        """
        The constructor just records relevant properties from the bubble
        profile.
        """
        self.first_field_name = bubble_profile.first_field.field_name
        self.fields_at_points = bubble_profile.fields_at_points
        lattice_configuration = (
             bubble_profile.spatial_lattice_configuration
        )
        self.spatial_step_in_inverse_GeV = (
            lattice_configuration.spatial_step_in_inverse_GeV
        )
        self.separation_character = separation_character

    def write_file(
            self,
            *,
            output_CSV_filename: str,
            solution_sample: SampleSet
    ):
        content_for_CSV = self.file_lines(solution_sample)
        with open(output_CSV_filename, "w") as output_file:
            output_file.write("\n".join(content_for_CSV) + "\n")

    def file_lines(self, solution_sample: SampleSet) -> List[str]:
        # TODO: enhance for second field
        return (
            [
                f"r in 1/GeV {self.separation_character}"
                f" {self.first_field_name} in GeV"
            ]
            + [
                self._data_row_as_string(
                    row_index=i,
                    fields_at_point=fields_at_point,
                    solution_sample=solution_sample
                )
                for i, fields_at_point in enumerate(self.fields_at_points)
            ]
        )

    def _data_row_as_string(
            self,
            *,
            row_index: int,
            fields_at_point: FieldCollectionAtPoint,
            solution_sample: SampleSet
    ) -> str:
        # TODO: enhance for second field
        radius_in_inverse_GeV = row_index * self.spatial_step_in_inverse_GeV
        first_field_strength = (
            fields_at_point.first_field.in_GeV(solution_sample)
        )
        return (
            f"{radius_in_inverse_GeV} {self.separation_character}"
            f" {first_field_strength}"
        )
