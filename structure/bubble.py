from typing import Optional
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
        domain_wall_alignment_weight = (
            2.0 * self.configuration.maximum_weight_difference
        )
        domain_end_weight = 2.0 * domain_wall_alignment_weight
        calculated_biases = BiasAccumulator()
        potential_values = (
            self.configuration.potential_in_quartic_GeV_per_field_step
        )
        spatial_step = self.configuration.spatial_step_in_inverse_GeV
        previous_profile = None
        for profile_at_point in self.fields_at_points:
            calculated_biases.add(
                profile_at_point.first_field.domain_wall_weights(
                    end_spin_weight=domain_end_weight,
                    spin_alignment_weight=domain_wall_alignment_weight
                )
            )
            calculated_biases.add(
                hamiltonian.potential.weights_for(
                    potential_in_quartic_GeV_per_field_step=potential_values,
                    single_field=profile_at_point.first_field
                )
            )
            if previous_profile:
                calculated_biases.add(
                    hamiltonian.kinetic.weights_for_difference(
                        at_smaller_radius=previous_profile.first_field,
                        at_larger_radius=profile_at_point.first_field,
                        radius_difference_in_inverse_GeV=spatial_step
                    )
                )
                if previous_profile.second_field:
                    calculated_biases.add(
                        hamiltonian.kinetic.weights_for_difference(
                            at_smaller_radius=previous_profile.second_field,
                            at_larger_radius=profile_at_point.second_field,
                            radius_difference_in_inverse_GeV=spatial_step
                        )
                    )
            previous_profile = profile_at_point
        return calculated_biases
