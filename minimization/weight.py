from typing import Dict, List, Optional, Tuple, TypeVar


SelfType = TypeVar("SelfType", bound="WeightAccumulator")
T = TypeVar("T")


def add_to_each(*, in_: List[float], from_: List[float]):
        for i, item_to_add in enumerate(from_):
            in_[i] += item_to_add


def scaled_linears_for_variable_names(
        *,
        variable_names: List[str],
        linear_weights: List[float],
        scaling_factor: float
) -> Dict[str, float]:
    return {
        n: (w * scaling_factor) for n, w in zip(variable_names, linear_weights)
        if w != 0.0
    }


class WeightAccumulator:
    linear_weights: Dict[str, float]
    quadratic_weights: Dict[Tuple[str, str], float]

    def __init__(
            self,
            *,
            linear_weights: Optional[Dict[str, float]] = None,
            quadratic_weights: Optional[Dict[Tuple[str, str], float]] = None
    ):
        self.linear_weights = linear_weights or {}
        self.quadratic_weights = quadratic_weights or {}

    def add_linears(self, weights_to_add: Dict[str, float]):
        for k, v in weights_to_add.items():
            self.linear_weights[k] = self.linear_weights.get(k, 0.0) + v

    def add_quadratics(self, weights_to_add: Dict[Tuple[str, str], float]):
        for k, v in weights_to_add.items():
            self.quadratic_weights[k] = self.quadratic_weights.get(k, 0.0) + v

    def add(self, other: SelfType):
        self.add_linears(other.linear_weights)
        self.add_quadratics(other.quadratic_weights)

    def for_QUBO(self) -> Dict[Tuple[str, str], float]:
        return {
            **self.quadratic_weights,
            **{(k, k): v for k, v in self.linear_weights.items()}
        }


class WeightTemplate:
    """
    This class encapsulates the aggregation of weights for linear and quadratic
    biases of spin or bit variables.
    IMPORTANT: Try to keep track of what field is the normal vector and what is
    the transpose vector, mainly for the sake of how the keys for the quadratic
    weights are created.
    IMPORTANT: Care has to be taken that the number of values given to __init__
    matches the number of variable names in the lists given to
    first_linears_for_variable_names,
    second_linears_for_variable_names, and
    quadratics_for_variable_names
    because there is no checking within these methods.
    """
    linear_weights_for_normal: List[float]
    linear_weights_for_transpose: List[float]
    quadratic_weights: List[List[float]]

    def __init__(
            self,
            *,
            # These are the number of variables there are for each FieldAtPoint.
            # (These may be the "first" and "second" fields of the model,
            # evaluated at the same spatial point, or they might be the same
            # field but at neighboring spatial points, or even the same field at
            # the same point, but for correlating its own variables with each
            # other.)
            number_of_values_for_normal: int,
            number_of_values_for_transpose: int,
            initial_quadratics: Optional[List[List[float]]] = None
    ):
        self.linear_weights_for_normal = [
            0.0 for _ in range(number_of_values_for_normal)
        ]
        self.linear_weights_for_transpose = [
            0.0 for _ in range(number_of_values_for_transpose)
        ]
        self.quadratic_weights = (
            initial_quadratics if initial_quadratics
            else [
                [0.0 for _ in range(number_of_values_for_normal)]
                for _ in range(number_of_values_for_transpose)
            ]
        )

    def add_linears_for_normal(self, weights_to_add: List[float]):
        """
        This increments self.linear_weights_for_normal by the all the elements
        in weights_to_add in order, without any guard against weights_to_add
        having too many elements for self.linear_weights_for_normal!
        """
        add_to_each(in_=self.linear_weights_for_normal, from_=weights_to_add)

    def add_linears_for_transpose(self, weights_to_add: List[float]):
        """
        This increments self.linear_weights_for_transpose by the all the
        elements in weights_to_add in order, without any guard against
        weights_to_add having too many elements for
        self.linear_weights_for_transpose!
        """
        add_to_each(in_=self.linear_weights_for_transpose, from_=weights_to_add)

    def add_quadratics(self, weights_to_add: List[List[float]]):
        """
        This increments self.quadratic_weights by the all the elements in
        weights_to_add in order, without any guard against weights_to_add having
        too many elements for self.quadratic_weights! The inner lists should
        match with a full normal vector for a single element of the transpose
        vector, and therefore the outer list should have the same length as the
        transpose vector.
        """
        for i, inner_list in enumerate(weights_to_add):
            for j, weight_to_add in enumerate(inner_list):
                self.quadratic_weights[i][j] += weight_to_add

    def normal_linears_for_names(
            self,
            *,
            variable_names: List[str],
            scaling_factor: float = 1.0
    ) -> Dict[str, float]:
        return scaled_linears_for_variable_names(
            variable_names=variable_names,
            linear_weights=self.linear_weights_for_normal,
            scaling_factor=scaling_factor
        )

    def transpose_linears_for_names(
            self,
            *,
            variable_names: List[str],
            scaling_factor: float = 1.0
    ) -> Dict[str, float]:
        return scaled_linears_for_variable_names(
            variable_names=variable_names,
            linear_weights=self.linear_weights_for_transpose,
            scaling_factor=scaling_factor
        )

    def quadratics_for_variable_names(
            self,
            *,
            normal_variable_names: List[str],
            transpose_variable_names: List[str],
            scaling_factor: float = 1.0
    ) -> Dict[str, float]:
        return {
            (s, f): (self.quadratic_weights[j][i] * scaling_factor)
            for i, f in enumerate(normal_variable_names)
            for j, s in enumerate(transpose_variable_names)
            if self.quadratic_weights[j][i] != 0.0
        }
