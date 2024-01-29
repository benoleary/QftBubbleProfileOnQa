from __future__ import annotations
from collections.abc import Sequence
from typing import Optional, Protocol

from basis.field import FieldAtPoint, FieldCollectionAtPoint
from minimization.weight import WeightAccumulator, WeightTemplate


class DomainWallWeighter(Protocol):
    """
    This class defines the methods which a BubbleProfile will use to obtain
    weights from Hamiltonian terms which set up the Ising-chain domain wall
    model for an annealer.
    """
    def weights_for_domain_walls(
            self,
            *,
            profiles_at_points: Sequence[FieldCollectionAtPoint],
            end_weight: float,
            alignment_weight: float
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

class TemplateDomainWallWeighter:
    """
    This class implements the methods of the DomainWallWeighter Protocol,
    provided that _set_alignment_weights_for_domain_wall has been overridden
    appropriately, using WeightTemplates for filling WeightAccumlators.
    """
    def weights_for_domain_walls(
            self,
            *,
            profiles_at_points: Sequence[FieldCollectionAtPoint],
            end_weight: float,
            alignment_weight: float
    ) -> WeightAccumulator:
        first_field_template, second_field_template = (
            self._domain_wall_templates_for_first_and_second_fields(
                fields_at_point=profiles_at_points[0],
                end_weight=end_weight,
                alignment_weight=alignment_weight
            )
        )

        domain_wall_weights = WeightAccumulator()
        for profile_point in profiles_at_points:
            domain_wall_weights.add(
                _domain_wall_weights_from_template(
                    field_template=first_field_template,
                    field_at_point=profile_point.first_field
                )
            )
            if second_field_template:
                domain_wall_weights.add(
                    _domain_wall_weights_from_template(
                        field_template=second_field_template,
                        field_at_point=profile_point.second_field
                    )
                )

        return domain_wall_weights

    def weights_for_fixed_value(
            self,
            *,
            field_at_point: FieldAtPoint,
            fixing_weight: float,
            number_of_ones: int
    ) -> WeightAccumulator:
        """
        This returns the weights to fix the spins of the given FieldAtPoint so
        that there are number_of_ones |1>s. Negative numbers can be given to
        instead specify -number_of_ones |0>s, similar to negative indices in a
        Python array.
        """
        if number_of_ones == 0:
            raise ValueError(
                "Input of 0 (or -0) should set all spins to |1> (or |0>) but"
                " this would prevent a domain wall"
            )
        number_of_values = field_at_point.field_definition.number_of_values
        if number_of_ones > number_of_values:
            raise ValueError(
                f"At most {number_of_values} can be set to |1>,"
                f" {number_of_ones} were requested"
            )
        if -number_of_ones > number_of_values:
            raise ValueError(
                f"At most {number_of_values} can be set to |0>,"
                f" {-number_of_ones} were requested (as negative input)"
            )
        return self._weights_for_fixed_value(
            variable_names=field_at_point.binary_variable_names,
            fixing_weight=fixing_weight,
            number_of_ones=number_of_ones
        )

    def _weights_for_fixed_value(
            self,
            *,
            variable_names: Sequence[str],
            fixing_weight: float,
            number_of_ones: int
    ) -> WeightAccumulator:
        """
        This sets up the weight now that number_of_ones has been validated.
        """
        raise NotImplementedError(
            "_weights_for_fixed_value needs to be overridden"
        )

    def _domain_wall_templates_for_first_and_second_fields(
            self,
            *,
            fields_at_point: FieldCollectionAtPoint,
            end_weight: float,
            alignment_weight: float
    ) -> tuple[WeightTemplate, Optional[WeightTemplate]]:
        first_field = fields_at_point.first_field
        first_field_template = self._weights_for_domain_wall(
            number_of_variables=(
                # For the purposes of weights for the Ising-chain domain wall
                # model, there are N + 1 variables to represent N values, and we
                # need to weight all N + 1 variables in this case.
                first_field.field_definition.number_of_values + 1
            ),
            end_weight=end_weight,
            alignment_weight=alignment_weight
        )

        second_field = fields_at_point.second_field
        second_field_template = (
            None if not second_field
            else self._weights_for_domain_wall(
                number_of_variables=(
                    # As above, we need to weight N + 1 variables.
                    second_field.field_definition.number_of_values + 1
                ),
                end_weight=end_weight,
                alignment_weight=alignment_weight
            )
        )
        return (first_field_template, second_field_template)

    def _weights_for_domain_wall(
            self,
            *,
            number_of_variables: int,
            end_weight: float,
            alignment_weight: float
    ) -> WeightTemplate:
        """
        This returns the weights to ensure that the spins are valid for the
        Ising-chain domain wall model, in a form that can be combined with a
        FieldAtPoint to create a pair of dicts in the form for sample_ising or
        sample_qubo, depending on the type of the variables.
        """
        # This is the case of a FieldAtPoint having correlations between its own
        # variables.
        domain_wall_weights = WeightTemplate(
            number_of_values_for_normal=number_of_variables,
            number_of_values_for_transpose=number_of_variables
        )

        # Now we need to apply the weights, which are rather different depending
        # on whether the variables are spin or bit.
        self._set_weights_for_domain_wall(
            domain_wall_weights=domain_wall_weights,
            number_of_variables=number_of_variables,
            end_weight=end_weight,
            alignment_weight=alignment_weight
        )
        return domain_wall_weights

    def _set_weights_for_domain_wall(
            self,
            *,
            domain_wall_weights: WeightTemplate,
            number_of_variables: int,
            end_weight: float,
            alignment_weight: float
    ):
        raise NotImplementedError(
            "_set_weights_for_domain_wall needs to be overridden"
        )


def _domain_wall_weights_from_template(
        *,
        field_template: WeightTemplate,
        field_at_point: FieldAtPoint
) -> WeightAccumulator:
    # In this case, the correlations are between the variables of the field with
    # other variables of the same field itself.
    return WeightAccumulator(
        linear_weights=field_template.normal_linears_for_names(
            variable_names=field_at_point.binary_variable_names
        ),
        quadratic_weights=field_template.quadratics_for_variable_names(
            normal_variable_names=field_at_point.binary_variable_names,
            transpose_variable_names=field_at_point.binary_variable_names
        )
    )
