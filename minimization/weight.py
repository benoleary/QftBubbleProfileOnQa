from typing import Dict, List, Optional, Tuple, TypeVar
from hamiltonian.field import FieldAtPoint


SelfType = TypeVar("SelfType", bound="WeightAccumulator")
T = TypeVar("T")

def add_to_each(*, in_: List[float], from_: List[float]):
        for i, item_to_add in enumerate(from_):
            in_[i] += item_to_add



class WeightAccumulator:
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
            second_number_of_values: int
    ):
        self.first_linear_weights = [0.0 for _ in range(first_number_of_values)]
        self.second_linear_weights = [
            0.0 for _ in range(second_number_of_values)
        ]
        self.quadratic_weights = [
            [0.0 for _ in range(first_number_of_values)]
            for _ in range(second_number_of_values)
        ]

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

    def add(self, other: SelfType):
        """
        This increments each weight of this WeightAccumulator by its
        corresponding weight in the other WeightAccumulator, without any check
        that they were created with the same shape! The other WeightAccumulator
        may have shorter lists, but must not have longer lists.
        """
        self.add_linears_for_first_field(other.first_linear_weights)
        self.add_linears_for_second_field(other.second_number_of_values)
        self.add_quadratics(other.quadratic_weights)

    def first_linears_for_variable_names(
            self,
            field_at_point: FieldAtPoint
    ) -> Dict[str, float]:
        return dict(
            zip(field_at_point.binary_variable_names, self.first_linear_weights)
        )

    def second_linears_for_variable_names(
            self,
            field_at_point: FieldAtPoint
    ) -> Dict[str, float]:
        return dict(
            zip(
                field_at_point.binary_variable_names,
                self.second_linear_weights
            )
        )

    def quadratics_for_variable_names(
            self,
            first_field: FieldAtPoint,
            second_field: FieldAtPoint
    ) -> Dict[str, float]:
        return {
            (f, s): self.quadratic_weights[j][i]
            for i, f in enumerate(first_field.binary_variable_names)
            for j, s in enumerate(second_field.binary_variable_names)
        }

class WeightDictContainer:
    def __init__(
            self,
            *,
            linear_weights: Dict[str, float],
            quadratic_weights: Dict[Tuple[str, str], float]
    ):
        self.linear_weights = linear_weights
        self.quadratic_weights = quadratic_weights

    def for_QUBO(self) -> Dict[Tuple[str, str], float]:
        return {
            **self.quadratic_weights,
            **{(k, k): v for k, v in self.linear_weights.items()}
        }