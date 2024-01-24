from typing import List, Optional, Tuple

from basis.field import FieldAtPoint, FieldCollectionAtPoint
from minimization.weight import WeightAccumulator, WeightTemplate


class SpinDomainWallWeighter:
    """
    This class implements the methods of the DomainWallWeighter Protocol,
    providing weights which should be used as input for the sample_ising method
    of a dimod.Sampler, as spin variables are assumed.
    """
    def weights_for_domain_walls(
            self,
            *,
            profiles_at_points: List[FieldCollectionAtPoint],
            end_weight: float,
            alignment_weight: float
    ) -> WeightAccumulator:
        first_field_template, second_field_template = (
            _domain_wall_templates_for_first_and_second_fields(
                profile_at_point=profiles_at_points[0],
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
        spin_biases = WeightAccumulator(
            initial_linears={
                binary_variable_name: fixing_weight
                for binary_variable_name
                in field_at_point.binary_variable_names[:number_of_ones]
            }
        )
        spin_biases.add_linears({
            binary_variable_name: -fixing_weight
            for binary_variable_name
            in field_at_point.binary_variable_names[number_of_ones:]
        })
        return spin_biases


def _domain_wall_templates_for_first_and_second_fields(
        *,
        fields_at_point: FieldCollectionAtPoint,
        end_weight: float,
        alignment_weight: float
) -> Tuple[WeightTemplate, Optional[WeightTemplate]]:
    first_field_template = _weights_for_domain_wall(
        number_of_spins=(
            fields_at_point.first_field.field_definition.number_of_values
        ),
        end_weight=end_weight,
        alignment_weight=alignment_weight
    )
    second_field_template = (
        None if not fields_at_point.second_field
        else _weights_for_domain_wall(
            number_of_spins=(
                fields_at_point.second_field.field_definition.number_of_values
            ),
            end_weight=end_weight,
            alignment_weight=alignment_weight
        )
    )
    return (first_field_template, second_field_template)


def _weights_for_domain_wall(
        *,
        number_of_spins: int,
        end_weight: float,
        alignment_weight: float
) -> WeightTemplate:
    """
    This returns the weights to ensure that the spins are valid for the
    Ising-chain domain wall model, in a form that can be combined with a
    FieldAtPoint to create a pair of dicts in the form for sample_ising: a dict
    of linear biases, which could be represented by a vector, and a dict of
    quadratic biases, which could be represented as an upper-triangular matrix
    of correlation weights, with zeros on the diagonal. (Apparently it is not
    necessary that the dict is "upper-triangular"; the middleware seems to
    cope.)
    """
    # This is the case of a FieldAtPoint having correlations between its own
    # variables.
    spin_weights = WeightTemplate(
        first_number_of_values=number_of_spins,
        second_number_of_values=number_of_spins
    )

    # First, we set the weights to fix the ends so that there is a domain of 1s
    # from the first index and a domain of 0s ending at the last index. The
    # signs are this way because we want the first spin to be |1> which
    # multiplies its weight by -1 in the objective function, and the last spin
    # to be |0> which multiplies its weight by +1.
    spin_weights.first_linear_weights[0] = end_weight
    spin_weights.first_linear_weights[-1] = -end_weight

    # Next, each pair of nearest neighbors gets weighted to favor having the
    # same values - which is either (-1)^2 or (+1)^2, so +1, while opposite
    # values multiply the weighting by (-1) * (+1) = -1. Therefore, a negative
    # weighting will penalize opposite spins with a positive contribution to the
    # objective function.
    for i in range(number_of_spins - 1):
        spin_weights.quadratic_weights[i][i + 1] = -alignment_weight

    return spin_weights


def _domain_wall_weights_from_template(
        *,
        field_template: WeightTemplate,
        field_at_point: FieldAtPoint
) -> WeightAccumulator:
    # In this case, the correlations are between the variables of the field with
    # other variables of the same field itself.
    return WeightAccumulator(
        linear_weights=field_template.first_linears_for_variable_names(
            field_at_point.binary_variable_names
        ),
        quadratic_weights=field_template.quadratics_for_variable_names(
            first_field=field_at_point,
            second_field=field_at_point
        )
    )
