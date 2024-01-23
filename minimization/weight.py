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
    }


class WeightAccumulator:
    def __init__(
            self,
            *,
            linear_weights: Dict[str, float],
            quadratic_weights: Dict[Tuple[str, str], float]
    ):
        self.linear_weights = linear_weights
        self.quadratic_weights = quadratic_weights

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
    """
    first_linear_weights: List[float]
    second_number_of_values: List[float]
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
            first_number_of_values: int,
            second_number_of_values: int,
            initial_quadratics: Optional[List[List[float]]] = None
    ):
        self.first_linear_weights = [0.0 for _ in range(first_number_of_values)]
        self.second_linear_weights = [
            0.0 for _ in range(second_number_of_values)
        ]
        self.quadratic_weights = (
            initial_quadratics if initial_quadratics
            else [
                [0.0 for _ in range(first_number_of_values)]
                for _ in range(second_number_of_values)
            ]
        )

    def add_linears_for_first_field(self, weights_to_add: List[float]):
        """
        This increments self.first_linear_weights by the all the elements in
        weights_to_add in order, without any guard against weights_to_add having
        too many elements for self.first_linear_weights!
        """
        add_to_each(in_=self.first_linear_weights, from_=weights_to_add)

    def add_linears_for_second_field(self, weights_to_add: List[float]):
        """
        This increments self.second_linear_weights by the all the elements in
        weights_to_add in order, without any guard against weights_to_add having
        too many elements for self.second_linear_weights!
        """
        add_to_each(in_=self.second_number_of_values, from_=weights_to_add)

    def add_quadratics(self, weights_to_add: List[List[float]]):
        """
        This increments self.quadratic_weights by the all the elements in
        weights_to_add in order, without any guard against weights_to_add having
        too many elements for self.quadratic_weights!
        """
        for i, inner_list in enumerate(weights_to_add):
            for j, weight_to_add in enumerate(inner_list):
                self.quadratic_weights[i][j] += weight_to_add

    def first_linears_for_variable_names(
            self,
            *,
            variable_names: List[str],
            scaling_factor: float = 1.0
    ) -> Dict[str, float]:
        return scaled_linears_for_variable_names(
            variable_names=variable_names,
            linear_weights=self.first_linear_weights,
            scaling_factor=scaling_factor
        )

    def second_linears_for_variable_names(
            self,
            *,
            variable_names: List[str],
            scaling_factor: float = 1.0
    ) -> Dict[str, float]:
        return scaled_linears_for_variable_names(
            variable_names=variable_names,
            linear_weights=self.second_linear_weights,
            scaling_factor=scaling_factor
        )

    def quadratics_for_variable_names(
            self,
            *,
            first_variable_names: List[str],
            second_variable_names: List[str],
            scaling_factor: float = 1.0
    ) -> Dict[str, float]:
        return {
            (f, s): (self.quadratic_weights[j][i] * scaling_factor)
            for i, f in enumerate(first_variable_names)
            for j, s in enumerate(second_variable_names)
        }
