from __future__ import annotations
from collections.abc import Sequence
from typing import Optional

from basis.field import FieldAtPoint, FieldCollectionAtPoint
from dynamics.hamiltonian import HasQftModelConfiguration
from input.configuration import QftModelConfiguration
from minimization.weight import WeightAccumulator, WeightTemplate


class BitHamiltonian(HasQftModelConfiguration):
    """
    This class implements the methods of the AnnealerHamiltonian Protocol,
    providing weights which should be used as input for the sample_qubo method
    of a dimod.Sampler, as bit variables are assumed.
    """
    def __init__(
            self,
            model_configuration: QftModelConfiguration
    ):
        super().__init__(model_configuration)
        self.model_configuration = model_configuration

        first_field = model_configuration.first_field
        second_field = model_configuration.second_field

        self.kinetic_weight_template_for_first_field = (
            _weights_for_single_field_kinetic_term_for_one_inverse_GeV_step(
                field_step_in_GeV=first_field.step_in_GeV,
                number_of_field_values=first_field.number_of_values
            )
        )
        self.kinetic_weight_template_for_second_field = (
            None if not second_field
            else (
                _weights_for_single_field_kinetic_term_for_one_inverse_GeV_step(
                    field_step_in_GeV=second_field.step_in_GeV,
                    number_of_field_values=second_field.number_of_values
                )
            )
        )

        self.potential_weight_template = (
            _weights_for_single_field_potential_at_point(
                model_configuration.potential_in_quartic_GeV_per_field_step[0]
            ) if not model_configuration.second_field
            else None # TODO: proper weights for two fields
        )

    def kinetic_weights(
            self,
            *,
            radius_step_in_inverse_GeV: float,
            nearer_center: FieldCollectionAtPoint,
            nearer_edge: FieldCollectionAtPoint,
            scaling_factor: float
    ) -> WeightAccumulator:
        scaling_including_spatial = (
            scaling_factor
            / (radius_step_in_inverse_GeV * radius_step_in_inverse_GeV)
        )

        kinetic_weights = _accumulator_from(
            weight_template=self.kinetic_weight_template_for_first_field,
            normal_variable_names=(
                nearer_center.first_field.binary_variable_names[1:-1]
            ),
            transpose_variable_names=(
                nearer_edge.first_field.binary_variable_names[1:-1]
            ),
            scaling_factor=scaling_including_spatial
        )

        second_template = self.kinetic_weight_template_for_second_field
        if second_template:
            kinetic_weights.add(
                 _accumulator_from(
                    weight_template=second_template,
                    normal_variable_names=(
                        nearer_center.second_field.binary_variable_names[1:-1]
                    ),
                    transpose_variable_names=(
                        nearer_edge.second_field.binary_variable_names[1:-1]
                    ),
                    scaling_factor=scaling_including_spatial
                )
            )

        return kinetic_weights

    def potential_weights(
            self,
            *,
            first_field: FieldAtPoint,
            second_field: Optional[FieldAtPoint] = None,
            scaling_factor: float
    ) -> WeightAccumulator:
        if not second_field:
            linear_weights = (
                self.potential_weight_template.normal_linears_for_names(
                    variable_names=first_field.binary_variable_names[1:-1],
                    scaling_factor=scaling_factor
                )
            )
            return WeightAccumulator(linear_weights=linear_weights)
        # TODO: do this properly
        raise NotImplementedError(
            "Not yet - use _weights_for_function_of_two_fields in __init__"
        )


def _weights_for_function_of_two_fields(
        function_value_matrix: Sequence[Sequence[float]]
) -> WeightTemplate:
    # For a function F(x, y), where x goes from 0 to N and y from 0 to M, the
    # weights for the discretization in the Ising-chain domain wall model for
    # bit variables are straightforward to calculate.
    number_of_values_for_transpose = len(function_value_matrix)
    first_row = function_value_matrix[0]
    number_of_values_for_normal = len(first_row)
    weight_template = WeightTemplate(
        number_of_values_for_normal=number_of_values_for_normal,
        number_of_values_for_transpose=number_of_values_for_transpose
    )
    # (Some pictures and some multi-line matrix representations would go a long
    # way in illuminating the logic, but it is too difficult to try to pull off
    # in UTF-8 comments...)
    # If the N (non-fixed) bit variables for x are arranged in a transposed
    # vector X^T = (x_1, x_2, ..., x_N), and Y is the same thing for y, we
    # define 2 weight vectors L_X and L_Y to "dot" with X and Y respectively (in
    # the sense that the first value of L_X will weight the first bit variable
    # of X), and a weight matrix W to "dot" between X^T and Y (so the i,j
    # element is the weight of the correlation of the ith bit variable of X with
    # the jth bit variable of Y).
    # However, we leave W for the moment, and look at L_X and L_Y. If x is 0
    # (all the non-fixed bit variables of X are 0), then we have the same form
    # as the case as described in _weights_for_single_field_potential_at_point,
    # so the ith weight of L_Y is (F(0, i) - F(0, (i - 1))), with i going from 1
    # to M (so the first weight is (F(0, 1) - F(0, 0)))). If y is j, meaning the
    # non-fixed bit variables at indices 1 to j are 1, then we have the sum of
    # all those differences, leading to a summed weight of (F(0, j) - F(0, 0)).
    # The value of F(0, 0) can be rolled into the overall constant of the
    # objective function which is irrelevant to the minimization anyway.
    for i in range(number_of_values_for_normal - 1):
        weight_template.linear_weights_for_normal[i] = (
            first_row[i + 1] - first_row[i]
        )
    for j in range(number_of_values_for_transpose - 1):
        weight_template.linear_weights_for_transpose[j] = (
            function_value_matrix[j + 1][0] - function_value_matrix[j][0]
        )

    # We define W as follows. The first row is (W_11, W_12, ..., W_1M) which
    # will "dot" with Y to then multiply with x_1. The second row is
    # (W_21, W_22, ..., W_2M), and so on, to the final row
    # (W_N1, W_N2, ..., W_NM).
    # We now use induction. If x has the value i and y the value j, and assuming
    # that the weights up to (x, y) = (i, (j - 1)) and to (x, y) = ((i - 1), j)
    # are correct (considering L_X and L_Y as well as W, and the weight of
    # (i, 0) is given by L_X_i, (0, j) by L_X_j and the weight of (i, j) by
    # W_ij), then the sum of weights of all (x, y) combinations from (0, 0) to
    # (i, (j - 1)) is F(i, (j - 1)). Likewise, all the combinations from (0, 0)
    # to ((i - 1), j) is F((i - 1), j). F(i, j) has to be the sum of all
    # combinations from (0, 0) to (i, j) which is all the combinations summing
    # to F(i, (j - 1)) plus all those summing to F((i - 1), j) but
    # double-counting the overlap, and that overlap sums to F((i - 1), (j - 1)).
    # Hence the weight of W_ij is
    # (F(i, j) - F((i - 1), j) - F(i, (j - 1)) + F((i - 1), (j - 1))).
    # It works for N = M = 2, so x can be 0 or 1 and y can be 0 or 1, so we need
    # only i = j = 1.
    # Both 0, we have 0 = F(0, 0) - F(0, 0).
    # For x = 1, y = 0, we have just L_X_1 = F(1, 0) - F(0, 0).
    # For x = 0, y = 1, we have just L_Y_1 = F(0, 1) - F(0, 0).
    # For x = 1, y = 1, we have L_X_1 + L_Y_1 + W_11 =
    # F(1, 0) - F(0, 0)
    # + F(0, 1) - F(0, 0)
    # + F(1, 1) - F(0, 1) - F(1, 0) + F(0, 0) = F(1, 1) - F(0, 0).
    # Explicitly going to x = 1, y = 2, we have
    # L_X_1 + L_Y_1 + L_Y_2 + W_11 + W_12 =
    # (L_X_1 + L_Y_1 + W_11) + L_Y_2 + W_12 =
    # F(1, 1) - F(0, 0)
    # + F(0, 2) - F(0, 1)
    # + F(1, 2) - F(1, 1) - F(0, 2) + F(0, 1) =
    # F(1, 2) - F(0, 0)
    for j in range(number_of_values_for_transpose - 1):
        for i in range(number_of_values_for_normal - 1):
            weight_template.quadratic_weights[j][i] = (
                function_value_matrix[j + 1][i + 1]
                - function_value_matrix[j][i + 1]
                - function_value_matrix[j + 1][i]
                + function_value_matrix[j][i]
            )

    return weight_template


def _accumulator_from(
        *,
        weight_template: WeightTemplate,
        normal_variable_names: Sequence[str],
        transpose_variable_names: Sequence[str],
        scaling_factor: float
) -> WeightAccumulator:
        return WeightAccumulator(
            linear_weights={
                **weight_template.normal_linears_for_names(
                    variable_names=normal_variable_names,
                    scaling_factor=scaling_factor
                ),
                **weight_template.transpose_linears_for_names(
                    variable_names=transpose_variable_names,
                    scaling_factor=scaling_factor
                )
            },
            quadratic_weights=(
                weight_template.quadratics_for_variable_names(
                    normal_variable_names=normal_variable_names,
                    transpose_variable_names=transpose_variable_names,
                    scaling_factor=scaling_factor
                )
            )
        )

def _weights_for_single_field_kinetic_term_for_one_inverse_GeV_step(
        *,
        field_step_in_GeV: float,
        number_of_field_values: int
) -> WeightTemplate:
        half_square_of_field_step = 0.5 * field_step_in_GeV * field_step_in_GeV
        return (
            _weights_for_function_of_two_fields(
                [
                    [
                        half_square_of_field_step * (x - y) * (x - y)
                        for y in range(number_of_field_values)
                    ]
                    for x in range(number_of_field_values)
                ]
            )
        )


def _weights_for_single_field_potential_at_point(
        potential_in_quartic_GeV_per_field_step: Sequence[float]
) -> WeightTemplate:
    """
    This function adds calculates weights for bit variables of a single field
    represented by a set of FieldAtPoint objects, one for every spatial point.
    (The calculation is significantly different when dealing with multiple
    fields at a single spatial point.)
    """
    # The weights are easier to work out than for the spin case, but they end up
    # with a similar form: if the nth non-fixed bit variable is |1>, then the
    # previous (n - 1) bit variables are also |1> so the objective function
    # already has the value for U(n - 1), so this bit variable must have a
    # weight U(n) - U(n - 1), with this sign because we are minimizing the
    # potential U (well, its integral combined with the integral of the kinetic
    # term) and the objective function is minimized.
    previous_value = potential_in_quartic_GeV_per_field_step[0]
    next_values = potential_in_quartic_GeV_per_field_step[1:]

    # This is the case of a single FieldAtPoint with no quadratic weights.
    weight_template = WeightTemplate(
        number_of_values_for_normal=len(next_values),
        number_of_values_for_transpose=0
    )

    for i, next_value in enumerate(next_values):
        # Compared to the spin case, the weights are twice as big and opposite
        # in sign.
        weight_template.linear_weights_for_normal[i] = (
            next_value - previous_value
        )
        previous_value = next_value

    return weight_template
