from typing import Optional
import minimization.variable
from configuration.configuration import DiscreteConfiguration
from hamiltonian.field import FieldAtPoint

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
        if second_field_name:
            self.second_field = FieldAtPoint(
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
