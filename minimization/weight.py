from typing import Dict, Optional, Tuple, TypeVar


SelfType = TypeVar("SelfType", bound="BiasAccumulator")
T = TypeVar("T")


def _copy_initial(
        *,
        source_mapping: Optional[Dict[T, float]],
        scaling_factor: float
    ) -> Dict[T, float]:
    if source_mapping is None:
        return {}
    return {k: (scaling_factor * v) for k, v in source_mapping.items()}


class BiasAccumulator:
    """
    This class encapsulates the aggregation of weights for linear and quadratic
    biases of spin variables.
    """
    def __init__(
            self,
            *,
            initial_linears: Dict[str, float] = None,
            initial_quadratics: Dict[Tuple[str, str], float] = None
        ):
        self.linear_biases = _copy_initial(
            source_mapping=initial_linears,
            scaling_factor=1.0
        )
        self.quadratic_biases = _copy_initial(
            source_mapping=initial_quadratics,
            scaling_factor=1.0
        )

    def add_linears(self, weights_to_add: Dict[str, float]):
        for k, v in weights_to_add.items():
            self.linear_biases[k] = self.linear_biases.get(k, 0.0) + v

    def add_quadratics(self, weights_to_add: Dict[Tuple[str, str], float]):
        for k, v in weights_to_add.items():
            self.quadratic_biases[k] = self.quadratic_biases.get(k, 0.0) + v

    def add(self, other: SelfType):
        self.add_linears(other.linear_biases)
        self.add_quadratics(other.quadratic_biases)

    def create_scaled_copy(self, scaling_factor: float) -> SelfType:
        return BiasAccumulator(
            initial_linears=_copy_initial(
                source_mapping=self.linear_biases,
                scaling_factor=scaling_factor
            ),
            initial_quadratics=_copy_initial(
                source_mapping=self.quadratic_biases,
                scaling_factor=scaling_factor
            )
        )
