from typing import List, Optional
import minimization.variable
from minimization.weight import BiasAccumulator
from configuration.configuration import DiscreteConfiguration
from hamiltonian.field import FieldAtPoint
import hamiltonian.potential
import hamiltonian.kinetic


class ProfileAtPoint:
    """
    This class represents the field or fields at a single point on the bubble
    profile.
    """
    def __init__(
            self,
            *,
            spatial_point_identifier: str,
            number_of_values_for_field: int,
            first_field_name: str,
            second_field_name: Optional[str] = None,
            field_step_in_GeV: float
        ):
        self.spatial_point_identifier = spatial_point_identifier
        self.first_field = FieldAtPoint(
                field_name=first_field_name,
                spatial_point_identifier=spatial_point_identifier,
                number_of_values_for_field=number_of_values_for_field,
                field_step_in_GeV=field_step_in_GeV
            )
        self.second_field = None if not second_field_name else FieldAtPoint(
                field_name=second_field_name,
                spatial_point_identifier=spatial_point_identifier,
                number_of_values_for_field=number_of_values_for_field,
                field_step_in_GeV=field_step_in_GeV
            )


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
        number_of_values_for_field = configuration.number_of_values_for_field
        spatial_name_function = minimization.variable.name_for_index(
            "r",
            configuration.number_of_spatial_steps
        )
        def create_profile_at_point(spatial_index: int) -> ProfileAtPoint:
            return ProfileAtPoint(
                spatial_point_identifier=spatial_name_function(spatial_index),
                number_of_values_for_field=number_of_values_for_field,
                first_field_name=configuration.first_field_name,
                second_field_name=configuration.second_field_name,
                field_step_in_GeV=configuration.field_step_in_GeV
            )
        self.fields_at_points = [
            create_profile_at_point(spatial_index=i)
            for i in range(configuration.number_of_spatial_steps + 1)
        ]
        self.spin_biases = self._set_up_weights()

    def _set_up_weights(self) -> BiasAccumulator:
        # We do the different terms in separate methods so that it is easier to
        # read.
        calculated_biases = self._get_model_weights(
            self._get_maximum_variable_weight()
        )
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
        maximum_field_difference = (
            self.configuration.field_step_in_GeV
            * (self.configuration.number_of_values_for_field - 1.0)
        )
        field_difference_over_radius_step = (
            maximum_field_difference
            / self.configuration.spatial_step_in_inverse_GeV
        )
        maximum_kinetic_term = (
            (1.0 if self.configuration.second_field_name else 0.5)
            * field_difference_over_radius_step
            * field_difference_over_radius_step
        )
        return (
            self.configuration.maximum_weight_difference + maximum_kinetic_term
        )

    def _get_model_weights(
            self,
            maximum_variable_weight: float
        ) -> BiasAccumulator:
        domain_wall_alignment_weight = 2.0 * maximum_variable_weight
        single_spin_fixing_weight = 2.0 * domain_wall_alignment_weight
        calculated_biases = self._get_fixed_center_and_edge_weights(
            single_spin_fixing_weight
        )

        for profile_at_point in self.fields_at_points[1:-1]:
            calculated_biases.add(
                profile_at_point.first_field.weights_for_domain_wall(
                    end_spin_weight=single_spin_fixing_weight,
                    spin_alignment_weight=domain_wall_alignment_weight
                )
            )

        return calculated_biases

    def _get_fixed_center_and_edge_weights(
            self,
            single_spin_fixing_weight: float
        ) -> BiasAccumulator:
        """
        This only works in the assumption of a single field which is set to
        1000... at the center and ...1110 at the edge, which is why the second
        field is ignored.
        """
        calculated_biases = (
            self.fields_at_points[0].first_field.weights_for_fixed_value(
                fixing_weight=single_spin_fixing_weight,
                number_of_down_spins=1
            )
        )
        calculated_biases.add(
            self.fields_at_points[-1].first_field.weights_for_fixed_value(
                fixing_weight=single_spin_fixing_weight,
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
