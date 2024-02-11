from __future__ import annotations
from collections.abc import Sequence

from dimod import SampleSet

from structure.bubble import BubbleProfile, ProfilePoint
from minimization.sampling import SampleProvider


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
        self.bubble_profile = bubble_profile
        lattice_configuration = (
             bubble_profile.spatial_lattice_configuration
        )
        self.spatial_step_in_inverse_GeV = (
            lattice_configuration.spatial_step_in_inverse_GeV
        )
        self.separation_character = separation_character

    def write_file_from_matrix(
            self,
            *,
            output_CSV_filename: str,
            header_row: Sequence[str],
            value_matrix: Sequence[Sequence[float]]
    ):
        content_for_CSV = (
            [self.separation_character.join(header_row)]
            + [
                self.separation_character.join([str(v) for v in value_row])
                for value_row in value_matrix
            ]
        )
        with open(output_CSV_filename, "w") as output_file:
            output_file.write("\n".join(content_for_CSV) + "\n")

    def write_file_from_sample(
            self,
            *,
            output_CSV_filename: str,
            solution_sample: SampleSet,
            sample_provider: SampleProvider
    ):
        content_for_CSV = self.file_lines(
            solution_sample=solution_sample,
            sample_provider=sample_provider
        )
        with open(output_CSV_filename, "w") as output_file:
            output_file.write("\n".join(content_for_CSV) + "\n")

    def file_lines(
            self,
            *,
            solution_sample: SampleSet,
            sample_provider: SampleProvider
    ) -> Sequence[str]:
        # TODO: enhance for second field
        profile_points = self.bubble_profile.field_strengths_at_radius_values(
            solution_sample=solution_sample,
            sample_provider=sample_provider
        )
        return (
            [
                f"r in 1/GeV {self.separation_character}"
                f" {self.bubble_profile.first_field.field_name} in GeV"
            ]
            + [
                self._data_row_as_string(profile_point)
                for profile_point in profile_points
            ]
        )

    def _data_row_as_string(self, profile_point: ProfilePoint) -> str:
        # TODO: enhance for second field
        return (
            f"{profile_point.radius_in_inverse_GeV} {self.separation_character}"
            f" {profile_point.first_field_strength_in_GeV}"
        )
