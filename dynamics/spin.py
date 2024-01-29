from __future__ import annotations
from collections.abc import Sequence
from typing import Optional

from basis.field import FieldDefinition, FieldCollectionAtPoint, FieldAtPoint
from dynamics.hamiltonian import HasQftModelConfiguration
from input.configuration import QftModelConfiguration
from minimization.weight import WeightAccumulator, WeightTemplate


class SpinHamiltonian(HasQftModelConfiguration):
    """
    This class implements the methods of the AnnealerHamiltonian Protocol,
    providing weights which should be used as input for the sample_ising method
    of a dimod.Sampler, as spin variables are assumed.
    """
    def __init__(
            self,
            model_configuration: QftModelConfiguration
    ):
        super().__init__(model_configuration)
        self.model_configuration = model_configuration

        self.first_field_positive_block = (
            _constant_quadratics_block_without_correlations_on_ends(
                model_configuration.first_field
            )
        )
        self.second_field_positive_block = (
            None if not model_configuration.second_field
            else _constant_quadratics_block_without_correlations_on_ends(
                model_configuration.second_field
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
        # We have to remember that this block ignores the fixed first |1> and
        # last |0>, so the lists of variable names have to be sliced
        # appropriately.
        scaling_including_spatial = (
            scaling_factor
            / (radius_step_in_inverse_GeV * radius_step_in_inverse_GeV)
        )

        kinetic_weights = _kinetic_weights_for_single_field(
            positive_block=self._positive_block_template_for(
                nearer_center.first_field.field_definition
            ),
            nearer_center=nearer_center.first_field,
            nearer_edge=nearer_edge.first_field,
            scaling_factor=scaling_including_spatial
        )

        if self.model_configuration.second_field:
            kinetic_weights.add(
                _kinetic_weights_for_single_field(
                    positive_block=self._positive_block_template_for(
                        nearer_center.second_field.field_definition
                    ),
                    nearer_center=nearer_center.second_field,
                    nearer_edge=nearer_edge.second_field,
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
        raise NotImplementedError("Not yet, but soon")

    def _positive_block_template_for(
            self,
            field_definition: FieldDefinition
    ) -> WeightTemplate:
        if field_definition == self.model_configuration.first_field:
            return self.first_field_positive_block
        if field_definition == self.model_configuration.second_field:
            return self.second_field_positive_block
        raise ValueError(f"unknown field {field_definition.field_name}")


def _kinetic_weights_for_single_field(
        *,
        positive_block: WeightTemplate,
        nearer_center: FieldAtPoint,
        nearer_edge: FieldAtPoint,
        scaling_factor: float
) -> WeightAccumulator:
    # Nearer center with itself
    kinetic_weights = WeightAccumulator(
        quadratic_weights=(
            positive_block.quadratics_for_variable_names(
                normal_variable_names=(
                    nearer_center.binary_variable_names[1:-1]
                ),
                transpose_variable_names=(
                    nearer_center.binary_variable_names[1:-1]
                ),
                scaling_factor=scaling_factor
            )
        )
    )
    # Nearer center with nearer edge
    kinetic_weights.add_quadratics(
        positive_block.quadratics_for_variable_names(
            normal_variable_names=nearer_center.binary_variable_names[1:-1],
            transpose_variable_names=(
                nearer_edge.binary_variable_names[1:-1]
            ),
            scaling_factor=-scaling_factor
        )
    )
    # Nearer edge with nearer center
    kinetic_weights.add_quadratics(
        positive_block.quadratics_for_variable_names(
            normal_variable_names=nearer_edge.binary_variable_names[1:-1],
            transpose_variable_names=(
                nearer_center.binary_variable_names[1:-1]
            ),
            scaling_factor=-scaling_factor
        )
    )
    # Nearer edge with itself
    kinetic_weights.add_quadratics(
        positive_block.quadratics_for_variable_names(
            normal_variable_names=nearer_edge.binary_variable_names[1:-1],
            transpose_variable_names=(
                nearer_edge.binary_variable_names[1:-1]
            ),
            scaling_factor=scaling_factor
        )
    )
    return kinetic_weights


def _weights_for_single_field_potential_at_point(
        potential_in_quartic_GeV_per_field_step: Sequence[float]
) -> WeightTemplate:
    """
    This function adds calculates weights for spin variables of a single field
    represented by a set of FieldAtPoint objects, one for every spatial point.
    (The calculation is significantly different when dealing with multiple
    fields at a single spatial point.)
    """
    # If we have 4 values U_0, U_1, U_2, and U_3, the field has 5 binary
    # variables, where the first and last are fixed, and there are 4 valid
    # bitstrings: 10000, 11000, 11100, and 11110. If the middle 3 spin variables
    # have weights A, B, and C respectively, then we have the following
    # simultaneous equations when considering the weights brought to the
    # objective function:
    # +A+B+C = U_0
    # -A+B+C = U_1
    # -A-B+C = U_2
    # -A-B-C = U_3
    # Therefore,
    # U_3 - U_2 = -2 C
    # U_2 - U_1 = -2 B
    # U_1 - U_0 = -2 A
    # and so we end up with the factor 0.5 and the potential for the lower index
    # minus the potential for the higher index.
    previous_value = potential_in_quartic_GeV_per_field_step[0]
    next_values = potential_in_quartic_GeV_per_field_step[1:]

    # This is the case of a single FieldAtPoint with no quadratic weights.
    weight_template = WeightTemplate(
        number_of_values_for_normal=len(next_values),
        number_of_values_for_transpose=0
    )

    for i, next_value in enumerate(next_values):
        weight_template.linear_weights_for_normal[i] = (
            0.5 * (previous_value - next_value)
        )
        previous_value = next_value

    return weight_template


def _weight_for_ACS_kinetic_term_for_one_inverse_GeV_step(
        field_step_in_GeV: float
) -> float:
    """
    This gives the weight which appears for every combination of spin variables
    between two FieldAtPoint instances as a positive contribution between those
    which both belong to the same FieldAtPoint and as a negative contribution
    between those which belong to different FieldAtPoint instances assuming a
    spatial step size of 1/GeV. Scaling for other spatial step sizes is
    straightforward.
    """
    # We assume that the field has the same step size at both spatial points.
    return (0.125 * field_step_in_GeV * field_step_in_GeV)


def _constant_quadratics_block_without_correlations_on_ends(
        field_definition: FieldDefinition
) -> WeightTemplate:
        # In this case, we have correlations between N - 1 spins because the
        # fixed first |1> and last |0> do not contribute to the square of the
        # difference of values.
        number_of_values = field_definition.number_of_values - 1
        kinetic_constant = (
            _weight_for_ACS_kinetic_term_for_one_inverse_GeV_step(
                field_definition.step_in_GeV
            )
        )
        return WeightTemplate(
            number_of_values_for_normal=0,
            number_of_values_for_transpose=0,
            initial_quadratics=[
                [kinetic_constant for _ in range(number_of_values)]
                for _ in range(number_of_values)
            ]
        )
