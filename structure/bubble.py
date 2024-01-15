from typing import Dict, List, Optional
from dimod import SampleSet
import minimization.sampling
import minimization.variable
from minimization.weight import BiasAccumulator
from configuration.configuration import DiscreteConfiguration
from hamiltonian.field import FieldAtPoint, FieldDefinition
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
            spatial_radius_in_inverse_GeV: float,
            first_field: FieldDefinition,
            second_field: Optional[FieldDefinition] = None
        ):
        self.spatial_point_identifier = spatial_point_identifier
        self.spatial_radius_in_inverse_GeV = spatial_radius_in_inverse_GeV
        self.first_field = FieldAtPoint(
            field_definition=first_field,
            spatial_point_identifier=spatial_point_identifier
        )
        self.second_field = None if not second_field else FieldAtPoint(
            field_definition=second_field,
            spatial_point_identifier=spatial_point_identifier
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
        spatial_name_function = minimization.variable.name_for_index(
            "r",
            configuration.number_of_spatial_steps
        )

        def create_profile_at_point(spatial_index: int) -> ProfileAtPoint:
            return ProfileAtPoint(
                spatial_point_identifier=spatial_name_function(spatial_index),
                spatial_radius_in_inverse_GeV=(
                    spatial_index * configuration.spatial_step_in_inverse_GeV
                ),
                first_field=configuration.first_field,
                second_field=configuration.second_field
            )

        self.number_of_point_profiles = (
            configuration.number_of_spatial_steps + 1
        )
        self.fields_at_points = [
            create_profile_at_point(spatial_index=i)
            for i in range(self.number_of_point_profiles)
        ]
        self.maximum_variable_weight = self._get_maximum_variable_weight()
        self.domain_wall_alignment_weight = 2.0 * self.maximum_variable_weight
        self.single_spin_fixing_weight = 2.0 * self.domain_wall_alignment_weight
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
                f.field_definition.field_name: f.in_GeV(lowest_energy_sample)
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
                f" {self.configuration.first_field.field_name} in GeV"
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
        first_field_kinetic = self._get_maximum_kinetic_for_single_field(
            self.configuration.first_field.step_in_GeV,
            self.configuration.number_of_values_for_first_field
        )
        maximum_kinetic = (
            first_field_kinetic if not self.configuration.second_field
            else (
                first_field_kinetic
                + self._get_maximum_kinetic_for_single_field(
                    self.configuration.second_field.step_in_GeV,
                    self.configuration.number_of_values_for_second_field
                )
            )
        )
        # The largest volume factor in our calculation is from the derivative
        # between the edge and its neighbor, so the volume factor at the edge is
        # an upper bound on the volume factor and that is sufficient.
        maximum_volume_factor = self._get_volume_factor(
            self.fields_at_points[-1].spatial_radius_in_inverse_GeV
        )
        return (
            maximum_kinetic
            + self.configuration.maximum_potential_weight_difference
        ) * maximum_volume_factor

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
        1000... (or as many 1s as implied by the true vacuum in steps according
        to the field definition object) at the center and ...1110 (or as many 0s
        as implied by the false vacuum in steps according to the field
        definition object) at the edge, which is why the second field is
        ignored.
        """
        center_first_field = self.fields_at_points[0].first_field
        first_field_definition = center_first_field.field_definition
        calculated_biases = (
            center_first_field.weights_for_fixed_value(
                fixing_weight=self.single_spin_fixing_weight,
                number_of_down_spins=(
                    1 + first_field_definition.true_vacuum_value_in_steps
                )
            )
        )
        calculated_biases.add(
            self.fields_at_points[-1].first_field.weights_for_fixed_value(
                fixing_weight=self.single_spin_fixing_weight,
                number_of_down_spins=(
                    1 + first_field_definition.false_vacuum_value_in_steps
                )
            )
        )
        return calculated_biases

    def _get_potential_weights(
            self,
            potential_values: List[List[float]]
        ) -> BiasAccumulator:
        calculated_biases = BiasAccumulator()
        # The fields at the bubble center and edge are fixed so it is not worth
        # evaluating the potential there.
        for profile_at_point in self.fields_at_points[1:-1]:
            without_volume_factor = hamiltonian.potential.weights_for(
                potential_in_quartic_GeV_per_field_step=potential_values[0],
                single_field=profile_at_point.first_field
            )
            volume_factor = self._get_volume_factor(
                profile_at_point.spatial_radius_in_inverse_GeV
            )
            calculated_biases.add(
                without_volume_factor.create_scaled_copy(volume_factor)
            )
        return calculated_biases

    def _get_kinetic_weights(
            self,
            spatial_step: float
        ) -> BiasAccumulator:
        calculated_biases = BiasAccumulator()
        previous_profile = self.fields_at_points[0]
        for profile_at_point in self.fields_at_points[1:]:
            # There are many ways to discretize a term which depends on the
            # radius and a derivative with respect to the radius. We take the
            # derivative at the half-way points of each interval, and use the
            # radius value for that half-way point.
            average_radius = 0.5 * (
                profile_at_point.spatial_radius_in_inverse_GeV
                + previous_profile.spatial_radius_in_inverse_GeV
            )
            volume_factor = self._get_volume_factor(average_radius)
            calculated_biases.add(
                hamiltonian.kinetic.weights_for_difference(
                    at_smaller_radius=previous_profile.first_field,
                    at_larger_radius=profile_at_point.first_field,
                    radius_difference_in_inverse_GeV=spatial_step
                ).create_scaled_copy(volume_factor)
            )
            if profile_at_point.second_field:
                calculated_biases.add(
                    hamiltonian.kinetic.weights_for_difference(
                        at_smaller_radius=previous_profile.second_field,
                        at_larger_radius=profile_at_point.second_field,
                        radius_difference_in_inverse_GeV=spatial_step
                    ).create_scaled_copy(volume_factor)
                )
            previous_profile = profile_at_point
        return calculated_biases

    def _get_volume_factor(self, radius_value: float) -> float:
        return radius_value**(self.configuration.volume_exponent)
