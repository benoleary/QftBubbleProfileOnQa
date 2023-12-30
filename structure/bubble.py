from typing import Dict, List, Optional
from dimod import SampleSet
import minimization.sampling
import minimization.variable
from minimization.weight import BiasAccumulator
from configuration.configuration import DiscreteConfiguration
from hamiltonian.field import FieldAtPoint
import minimization.sampling
import hamiltonian.kinetic
import hamiltonian.potential


_separation_character = ";"


class ProfileAtPoint:
    """
    This class represents the field or fields at a single point on the bubble
    profile.
    """
    def __init__(
            self,
            *,
            spatial_point_identifier: str,
            first_field_name: str,
            first_field_step_in_GeV: float,
            first_field_offset_in_GeV: float,
            number_of_values_for_first_field: int,
            second_field_name: Optional[str] = None,
            second_field_step_in_GeV: Optional[float] = None,
            second_field_offset_in_GeV: Optional[float] = None,
            number_of_values_for_second_field: int
        ):
        self.spatial_point_identifier = spatial_point_identifier
        self.first_field = FieldAtPoint(
                field_name=first_field_name,
                spatial_point_identifier=spatial_point_identifier,
                number_of_values_for_field=number_of_values_for_first_field,
                field_step_in_GeV=first_field_step_in_GeV,
                offset_from_origin_in_GeV=first_field_offset_in_GeV
            )
        self.second_field = None if not second_field_name else FieldAtPoint(
                field_name=second_field_name,
                spatial_point_identifier=spatial_point_identifier,
                number_of_values_for_field=number_of_values_for_second_field,
                field_step_in_GeV=second_field_step_in_GeV,
                offset_from_origin_in_GeV=second_field_offset_in_GeV
            )

    def get_fields(self) -> List[FieldAtPoint]:
        if not self.second_field:
            return [self.first_field]
        return [self.first_field, self.second_field]


class BubbleProfile:
    """
    This class represents the field or fields at all the points of the bubble
    profile.
    """
    def __init__(self, configuration: DiscreteConfiguration):
        """
        The constructor just sets up fields.
        """
        self.configuration = configuration
        self.maximum_variable_weight = self._get_maximum_variable_weight()
        self.domain_wall_alignment_weight = 2.0 * self.maximum_variable_weight
        self.single_spin_fixing_weight = 2.0 * self.domain_wall_alignment_weight
        spatial_name_function = minimization.variable.name_for_index(
            "r",
            configuration.number_of_spatial_steps
        )

        def create_profile_at_point(spatial_index: int) -> ProfileAtPoint:
            return ProfileAtPoint(
                spatial_point_identifier=spatial_name_function(spatial_index),
                first_field_name=configuration.first_field_name,
                first_field_step_in_GeV=configuration.first_field_step_in_GeV,
                first_field_offset_in_GeV=(
                    configuration.first_field_offset_in_GeV
                ),
                number_of_values_for_first_field=(
                    configuration.number_of_values_for_first_field
                ),
                second_field_name=configuration.second_field_name,
                second_field_step_in_GeV=configuration.second_field_step_in_GeV,
                second_field_offset_in_GeV=(
                    configuration.second_field_offset_in_GeV
                ),
                number_of_values_for_second_field=(
                    configuration.number_of_values_for_second_field
                )
            )

        self.number_of_point_profiles = (
            configuration.number_of_spatial_steps + 1
        )
        self.fields_at_points = [
            create_profile_at_point(spatial_index=i)
            for i in range(self.number_of_point_profiles)
        ]
        self.spin_biases = self._set_up_weights()

    def map_radius_labels_to_field_strengths_from_lowest_sample(
            self,
            sample_set: SampleSet
        ) -> Dict[str, Dict[str, float]]:
        lowest_energy_sample = minimization.sampling.get_lowest_sample_from_set(
            sample_set
        )
        return {
            p.spatial_point_identifier: {
                f.field_name: f.in_GeV(lowest_energy_sample)
                for f in p.get_fields()
            }
            for p in self.fields_at_points
        }

    def lowest_sample_as_CSV_file_content(
            self,
            sample_set: SampleSet
        ) -> List[str]:
        lowest_energy_sample = minimization.sampling.get_lowest_sample_from_set(
            sample_set
        )
        return (
            [
                f"r in 1/GeV {_separation_character}"
                f" {self.configuration.first_field_name} in GeV"
            ]
            + [
                self._row_for_CSV(row_index=r, sample_set=lowest_energy_sample)
                for r in range(self.number_of_point_profiles)
            ]
        )

    def _row_for_CSV(
            self,
            *,
            row_index: int,
            sample_set: SampleSet
        ) -> str:
        radius_in_inverse_GeV = (
            row_index * self.configuration.spatial_step_in_inverse_GeV
        )
        first_field_strength = (
            self.fields_at_points[row_index].first_field.in_GeV(sample_set)
        )
        return (
            f"{radius_in_inverse_GeV} {_separation_character}"
            f" {first_field_strength}"
        )

    def _set_up_weights(self) -> BiasAccumulator:
        # We do the different terms in separate methods so that it is easier to
        # read.
        calculated_biases = self._get_model_weights()
        calculated_biases.add(
            self._get_potential_weights(
                self.configuration.potential_in_quartic_GeV_per_field_step
            )
        )
        calculated_biases.add(
            self._get_kinetic_weights(
                self.configuration.spatial_step_in_inverse_GeV
            )
        )
        return calculated_biases

    def _get_maximum_variable_weight(self) -> float:
        first_contribution = self._get_maximum_kinetic_for_single_field(
            self.configuration.first_field_step_in_GeV,
            self.configuration.number_of_values_for_first_field
        )
        if not self.configuration.second_field_name:
            return (
                first_contribution
                + self.configuration.maximum_weight_difference
            )
        return (
            first_contribution
            + self._get_maximum_kinetic_for_single_field(
                self.configuration.second_field_step_in_GeV,
                self.configuration.number_of_values_for_second_field
            )
            + self.configuration.maximum_weight_difference
        )

    def _get_maximum_kinetic_for_single_field(
            self,
            field_step_in_GeV: float,
            number_of_values_for_field: int
        ) -> float:
        maximum_field_difference = (
            field_step_in_GeV * (number_of_values_for_field - 1.0)
        )
        field_difference_over_radius_step = (
            maximum_field_difference
            / self.configuration.spatial_step_in_inverse_GeV
        )
        return (
            0.5
            * field_difference_over_radius_step
            * field_difference_over_radius_step
        )

    def _get_model_weights(self) -> BiasAccumulator:
        calculated_biases = self._get_fixed_center_and_edge_weights()

        for profile_at_point in self.fields_at_points[1:-1]:
            calculated_biases.add(
                profile_at_point.first_field.weights_for_domain_wall(
                    end_spin_weight=self.single_spin_fixing_weight,
                    spin_alignment_weight=self.domain_wall_alignment_weight
                )
            )

        return calculated_biases

    def _get_fixed_center_and_edge_weights(self) -> BiasAccumulator:
        """
        This only works in the assumption of a single field which is set to
        1000... at the center and ...1110 at the edge, which is why the second
        field is ignored.
        """
        calculated_biases = (
            self.fields_at_points[0].first_field.weights_for_fixed_value(
                fixing_weight=self.single_spin_fixing_weight,
                number_of_down_spins=1
            )
        )
        calculated_biases.add(
            self.fields_at_points[-1].first_field.weights_for_fixed_value(
                fixing_weight=self.single_spin_fixing_weight,
                number_of_down_spins=-1
            )
        )

        return calculated_biases

    def _get_potential_weights(
            self,
            potential_values: List[float]
        ) -> BiasAccumulator:
        calculated_biases = BiasAccumulator()
        # The fields at the bubble center and edge are fixed so it is not worth
        # evaluating the potential there.
        for profile_at_point in self.fields_at_points[1:-1]:
            calculated_biases.add(
                hamiltonian.potential.weights_for(
                    potential_in_quartic_GeV_per_field_step=potential_values,
                    single_field=profile_at_point.first_field
                )
            )
        return calculated_biases

    def _get_kinetic_weights(
            self,
            spatial_step: float
        ) -> BiasAccumulator:
        calculated_biases = BiasAccumulator()
        previous_profile = self.fields_at_points[0]
        for profile_at_point in self.fields_at_points[1:]:
            calculated_biases.add(
                hamiltonian.kinetic.weights_for_difference(
                    at_smaller_radius=previous_profile.first_field,
                    at_larger_radius=profile_at_point.first_field,
                    radius_difference_in_inverse_GeV=spatial_step
                )
            )
            if profile_at_point.second_field:
                calculated_biases.add(
                    hamiltonian.kinetic.weights_for_difference(
                        at_smaller_radius=previous_profile.second_field,
                        at_larger_radius=profile_at_point.second_field,
                        radius_difference_in_inverse_GeV=spatial_step
                    )
                )
            previous_profile = profile_at_point
        return calculated_biases
