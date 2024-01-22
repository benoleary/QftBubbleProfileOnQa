from typing import List, Optional
from hamiltonian.field import FieldAtPoint, FieldDefinition


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
