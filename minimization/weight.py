from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Optional, TypeVar


SelfType = TypeVar("SelfType", bound="WeightAccumulator")
T = TypeVar("T")


def add_to_each(*, in_: Sequence[float], from_: Iterable[float]):
        for i, item_to_add in enumerate(from_):
            in_[i] += item_to_add


def scaled_linears_for_variable_names(
        *,
        variable_names: Iterable[str],
        linear_weights: Iterable[float],
        scaling_factor: float
) -> dict[str, float]:
    return {
        n: (w * scaling_factor) for n, w in zip(variable_names, linear_weights)
        if w != 0.0
    }


class WeightAccumulator:
    linear_weights: dict[str, float]
    quadratic_weights: dict[tuple[str, str], float]

    def __init__(
            self,
            *,
            linear_weights: Optional[dict[str, float]] = None,
            quadratic_weights: Optional[dict[tuple[str, str], float]] = None
    ):
        self.linear_weights = linear_weights or {}
        self.quadratic_weights = quadratic_weights or {}

    def add_linears(self, weights_to_add: dict[str, float]):
        for k, v in weights_to_add.items():
            self.linear_weights[k] = self.linear_weights.get(k, 0.0) + v

    def add_quadratics(self, weights_to_add: dict[tuple[str, str], float]):
        for k, v in weights_to_add.items():
            self.quadratic_weights[k] = self.quadratic_weights.get(k, 0.0) + v

    def add(self, other: SelfType):
        self.add_linears(other.linear_weights)
        self.add_quadratics(other.quadratic_weights)

    def for_QUBO(self) -> dict[tuple[str, str], float]:
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
    linear_weights_for_normal: Sequence[float]
    linear_weights_for_transpose: Sequence[float]
    quadratic_weights: Sequence[Sequence[float]]

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
            initial_quadratics: Optional[Sequence[Sequence[float]]] = None
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

    def normal_linears_for_names(
            self,
            *,
            variable_names: Iterable[str],
            scaling_factor: float = 1.0
    ) -> dict[str, float]:
        return scaled_linears_for_variable_names(
            variable_names=variable_names,
            linear_weights=self.linear_weights_for_normal,
            scaling_factor=scaling_factor
        )

    def transpose_linears_for_names(
            self,
            *,
            variable_names: Iterable[str],
            scaling_factor: float = 1.0
    ) -> dict[str, float]:
        return scaled_linears_for_variable_names(
            variable_names=variable_names,
            linear_weights=self.linear_weights_for_transpose,
            scaling_factor=scaling_factor
        )

    def quadratics_for_variable_names(
            self,
            *,
            normal_variable_names: Iterable[str],
            transpose_variable_names: Iterable[str],
            scaling_factor: float = 1.0
    ) -> dict[str, float]:
        return {
            (t_n, n_n): (self.quadratic_weights[t_i][n_i] * scaling_factor)
            for n_i, n_n in enumerate(normal_variable_names)
            for t_i, t_n in enumerate(transpose_variable_names)
            if self.quadratic_weights[t_i][n_i] != 0.0
        }
