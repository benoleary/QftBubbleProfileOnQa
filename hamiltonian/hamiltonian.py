from typing import Optional, Protocol
from hamiltonian.field import FieldAtPoint
from minimization.weight import WeightAccumulator

class AnnealerHamiltonian(Protocol):
    """
    This class defines the methods which a BubbleProfile will use to obtain
    weights for an annealer.
    """
    def domain_wall_weights(
            self,
            field_at_point: FieldAtPoint
    ) -> WeightAccumulator:
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
