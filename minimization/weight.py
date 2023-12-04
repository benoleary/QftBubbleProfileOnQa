from typing import Dict, Tuple

class AccumulatingWeightMatrix:
    """
    This class encapsulates the aggregation of weights for pairs of binary
    variables.
    """
    def __init__(self, initial_weights: Dict[Tuple[str, str], float]):
        self.weight_matrix = {k: v for k, v in initial_weights.items()}

    def add(self, weights_to_add: Dict[Tuple[str, str], float]):
        for k, v in weights_to_add.items():
            self.weight_matrix[k] = self.weight_matrix.get(k, 0.0) + v
