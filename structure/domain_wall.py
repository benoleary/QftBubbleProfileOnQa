from typing import Protocol
from hamiltonian.field import FieldAtPoint
from minimization.weight import WeightAccumulator

class DomainWallWeighter(Protocol):
    """
    This class defines the methods which a BubbleProfile will use to obtain
    weights from Hamiltonian terms which set up the Ising-chain domain wall
    model for an annealer.
    """
    def weights_for_domain_wall(
            self,
            *,
            field_at_point: FieldAtPoint,
            end_spin_weight: float,
            spin_alignment_weight: float
    ) -> WeightAccumulator:
        raise NotImplementedError("DomainWallWeighter is just a Protocol")

    def weights_for_fixed_value(
         self,
            *,
            field_at_point: FieldAtPoint,
            fixing_weight: float,
            number_of_ones: int
    ) -> WeightAccumulator:
        """
        This should return the weights to fix the spins or bits of the given
        FieldAtPoint so that there are number_of_ones |1>s. Negative  numbers
        can be given to instead specify -number_of_ones |0>s, similar to
        negative indices in a Python array.
        """
        raise NotImplementedError("DomainWallWeighter is just a Protocol")
