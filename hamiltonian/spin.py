from typing import List, Optional
from hamiltonian.field import FieldDefinition, FieldAtPoint
from minimization.weight import WeightAccumulator, WeightTemplate
from configuration.configuration import QftModelConfiguration


def _weights_for_domain_wall(
        *,
        number_of_spins: int,
        end_spin_weight: float,
        spin_alignment_weight: float
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
    spin_weights.first_linear_weights[0] = end_spin_weight
    spin_weights.first_linear_weights[-1] = -end_spin_weight

    # Next, each pair of nearest neighbors gets weighted to favor having the
    # same values - which is either (-1)^2 or (+1)^2, so +1, while opposite
    # values multiply the weighting by (-1) * (+1) = -1. Therefore, a negative
    # weighting will penalize opposite spins with a positive contribution to the
    # objective function.
    for i in range(number_of_spins - 1):
        spin_weights.quadratic_weights[i][i + 1] = -spin_alignment_weight

    return spin_weights


def _weights_for_single_field_potential_at_point(
        potential_in_quartic_GeV_per_field_step: List[float]
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
    spin_weights = WeightTemplate(
        first_number_of_values=len(next_values),
        second_number_of_values=0
    )

    for i, next_value in enumerate(next_values):
        spin_weights[i] = 0.5 * (previous_value - next_value)

    return spin_weights


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


def _constant_quadratics_block(
        field_definition: FieldDefinition
) -> WeightTemplate:
        number_of_values = field_definition.number_of_values
        kinetic_constant = (
            _weight_for_ACS_kinetic_term_for_one_inverse_GeV_step(
                field_definition.step_in_GeV
            )
        )
        return  WeightTemplate(
            first_number_of_values=0,
            second_number_of_values=0,
            initial_quadratics=[
                [kinetic_constant for _ in range(number_of_values)]
                for _ in range(number_of_values)
            ]
        )


class SpinHamiltonian:
    """
    This class implements the methods of the AnnealerHamiltonian Protocol,
    providing weights which should be used as input for the sample_ising method
    of a dimod.Sampler, as spin variables are assumed.
    """
    def __init__(
            self,
            model_configuration: QftModelConfiguration
    ):
        self.model_configuration = model_configuration
        self.first_field_domain_wall_template = _weights_for_domain_wall(
            model_configuration.first_field.number_of_values + 1
        )
        self.second_field_domain_wall_template = (
            None if not  model_configuration.second_field
            else _weights_for_domain_wall(
                model_configuration.second_field.number_of_values + 1
            )
        )
        self.first_field_positive_block = _constant_quadratics_block(
            model_configuration.first_field
        )
        self.second_field_positive_block = (
            None if not model_configuration.second_field
            else _constant_quadratics_block(model_configuration.second_field)
        )
        self.potential_weight_template = (
            _weights_for_single_field_potential_at_point(
                model_configuration.potential_in_quartic_GeV_per_field_step[0]
            ) if not model_configuration.second_field
            else None # TODO: proper weights for two fields
        )

    def domain_wall_weights(
            self,
            field_at_point: FieldAtPoint
    ) -> WeightAccumulator:
        weight_template = self._linear_weight_template_for(
            field_at_point.field_definition
        )
        return WeightAccumulator(
            linear_weights=weight_template.first_linears_for_variable_names(
                field_at_point
            ),
            quadratic_weights=weight_template.quadratics_for_variable_names(
                first_field=field_at_point,
                second_field=field_at_point
            )
        )

    def kinetic_weights(
            self,
            *,
            radius_step_in_inverse_GeV: float,
            nearer_center: FieldAtPoint,
            nearer_edge: FieldAtPoint,
            scaling_factor: float
    ) -> WeightAccumulator:
        positive_block = self._positive_block_template_for (
            nearer_center.field_definition
        )
        scaling_including_spatial = (
            scaling_factor
            / (radius_step_in_inverse_GeV * radius_step_in_inverse_GeV)
        )
        # Nearer center with itself
        kinetic_weights = WeightAccumulator(
            quadratic_weights=(
                positive_block.quadratics_for_variable_names(
                    first_field=nearer_center,
                    second_field=nearer_center,
                    scaling_factor=scaling_including_spatial
                )
            )
        )
        # Nearer center with nearer edge
        kinetic_weights.add_quadratics(
            positive_block.quadratics_for_variable_names(
                first_field=nearer_center,
                second_field=nearer_edge,
                scaling_factor=-scaling_including_spatial
            )
        )
        # Nearer edge with nearer center
        kinetic_weights.add_quadratics(
            positive_block.quadratics_for_variable_names(
                first_field=nearer_edge,
                second_field=nearer_center,
                scaling_factor=-scaling_including_spatial
            )
        )
        # Nearer edge with itself
        kinetic_weights.add_quadratics(
            positive_block.quadratics_for_variable_names(
                first_field=nearer_edge,
                second_field=nearer_edge,
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
            return (
                self.potential_weight_template.first_linears_for_variable_names(
                    field_at_point=first_field,
                    scaling_factor=scaling_factor
                )
            )
        # TODO: do this properly
        return WeightAccumulator(
            linear_weights={},
            quadratic_weights={}
        )

    def _template_for_first_or_second(
            self,
            *,
            field_definition: FieldDefinition,
            template_for_first: WeightTemplate,
            template_for_second: WeightTemplate
    ) -> WeightTemplate:
        if field_definition == self.model_configuration.first_field:
            return template_for_first
        if field_definition == self.model_configuration.second_field:
            return template_for_second
        raise ValueError(f"unknown field {field_definition.field_name}")

    def _linear_weight_template_for(
            self,
            field_definition: FieldDefinition
    ) -> WeightTemplate:
        return self._template_for_first_or_second(
            field_definition=field_definition,
            template_for_first=self.first_field_domain_wall_template,
            template_for_second=self.second_field_domain_wall_template
        )

    def _positive_block_template_for(
            self,
            field_definition: FieldDefinition
    ) -> WeightTemplate:
        return self._template_for_first_or_second(
            field_definition=field_definition,
            template_for_first=self.first_field_positive_block,
            template_for_second=self.second_field_positive_block
        )
