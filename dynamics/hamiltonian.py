from __future__ import annotations
from typing import Optional, Protocol

from basis.field import FieldAtPoint, FieldCollectionAtPoint, FieldDefinition
from input.configuration import QftModelConfiguration
from minimization.weight import WeightAccumulator


class AnnealerHamiltonian(Protocol):
    """
    This class defines the methods which a BubbleProfile will use to obtain
    weights from a QFT Hamiltonian for an annealer.
    """
    def get_first_field_definition(self) -> FieldDefinition:
        raise NotImplementedError("AnnealerHamiltonian is just a Protocol")

    def get_second_field_definition(self) -> Optional[FieldDefinition]:
        raise NotImplementedError("AnnealerHamiltonian is just a Protocol")

    def get_maximum_kinetic_contribution(
            self,
            radius_step_in_inverse_GeV: float
    ) -> float:
        raise NotImplementedError("AnnealerHamiltonian is just a Protocol")

    def get_maximum_potential_difference(self) -> float:
        raise NotImplementedError("AnnealerHamiltonian is just a Protocol")

    def kinetic_weights(
            self,
            *,
            radius_step_in_inverse_GeV: float,
            nearer_center: FieldCollectionAtPoint,
            nearer_edge: FieldCollectionAtPoint,
            scaling_factor: float
    ) -> WeightAccumulator:
        raise NotImplementedError("AnnealerHamiltonian is just a Protocol")

    def potential_weights(
            self,
            *,
            first_field: FieldAtPoint,
            second_field: Optional[FieldAtPoint] = None,
            scaling_factor: float
    ) -> WeightAccumulator:
        raise NotImplementedError("AnnealerHamiltonian is just a Protocol")


class HasQftModelConfiguration:
    def __init__(
            self,
            model_configuration: QftModelConfiguration
    ):
        self.model_configuration = model_configuration
        potential_rows = (
            model_configuration.potential_in_quartic_GeV_per_field_step
        )

        self.minimum_potential = potential_rows[0][0]
        self.maximum_potential = self.minimum_potential
        for potential_row in potential_rows:
            for potential_value in potential_row:
                if potential_value < self.minimum_potential:
                    self.minimum_potential = potential_value
                if potential_value > self.maximum_potential:
                    self.maximum_potential = potential_value
        self.maximum_potential_difference = (
            self.maximum_potential - self.minimum_potential
        )

    def get_first_field_definition(self) -> FieldDefinition:
        return self.model_configuration.first_field

    def get_second_field_definition(self) -> Optional[FieldDefinition]:
        return self.model_configuration.second_field

    def get_maximum_kinetic_contribution(
            self,
            radius_step_in_inverse_GeV: float
    ) -> float:
        first_field_kinetic = _get_maximum_kinetic_for_single_field(
            field_definition=self.model_configuration.first_field,
            radius_step_in_inverse_GeV=radius_step_in_inverse_GeV
        )
        if not self.model_configuration.second_field:
            return first_field_kinetic

        return (
            first_field_kinetic
            + _get_maximum_kinetic_for_single_field(
                field_definition=self.model_configuration.second_field,
                radius_step_in_inverse_GeV=radius_step_in_inverse_GeV
            )
        )

    def get_maximum_potential_difference(self) -> float:
        return self.maximum_potential_difference


def _get_maximum_kinetic_for_single_field(
        *,
        field_definition: FieldDefinition,
        radius_step_in_inverse_GeV: float
) -> float:
    maximum_field_difference = (
        field_definition.step_in_GeV
        * (field_definition.number_of_values - 1.0)
    )
    field_difference_over_radius_step = (
        maximum_field_difference / radius_step_in_inverse_GeV
    )
    return (
        0.5
        * field_difference_over_radius_step
        * field_difference_over_radius_step
    )
