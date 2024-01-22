from typing import List, Optional, Protocol
from hamiltonian.field import FieldAtPoint, FieldDefinition
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
            nearer_center: FieldAtPoint,
            nearer_edge: FieldAtPoint,
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


class HasDiscretizedPotential:
    def __init__(
            self,
            potential_in_quartic_GeV_per_field_step: List[List[float]]
    ):
        self.potential_in_quartic_GeV_per_field_step = (
            potential_in_quartic_GeV_per_field_step
        )

        self.minimum_potential = potential_in_quartic_GeV_per_field_step[0][0]
        self.maximum_potential = self.minimum_potential
        for potential_row in self.potential_in_quartic_GeV_per_field_step:
            for potential_value in potential_row:
                if potential_value < self.minimum_potential:
                    self.minimum_potential = potential_value
                if potential_value > self.maximum_potential:
                    self.maximum_potential = potential_value
        self.maximum_potential_difference = (
            self.maximum_potential - self.minimum_potential
        )