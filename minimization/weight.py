from typing import Dict, Optional, Tuple, TypeVar

SelfType = TypeVar("SelfType", bound="BiasAccumulator")
T = TypeVar("T")

def _copy_initial(source: Optional[Dict[T, float]]) -> Dict[T, float]:
    if source is None:
        return {}
    return {k: v for k, v in source.items()}


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
        self.linear_biases = _copy_initial(initial_linears)
        self.quadratic_biases = _copy_initial(initial_quadratics)

    def add_linears(self, weights_to_add: Dict[str, float]):
        for k, v in weights_to_add.items():
            self.linear_biases[k] = self.linear_biases.get(k, 0.0) + v

    def add_quadratics(self, weights_to_add: Dict[Tuple[str, str], float]):
        for k, v in weights_to_add.items():
            self.quadratic_biases[k] = self.linear_biases.get(k, 0.0) + v

    def add(self, other: SelfType):
        self.add_linears(other.linear_biases)
        self.add_quadratics(other.quadratic_biases)
