from __future__ import annotations
from collections.abc import Sequence

from minimization.weight import WeightAccumulator, WeightTemplate
from structure.domain_wall import TemplateDomainWallWeighter


class SpinDomainWallWeighter(TemplateDomainWallWeighter):
    """
    This class implements the methods of the DomainWallWeighter Protocol,
    providing weights which should be used as input for the sample_ising method
    of a dimod.Sampler, as spin variables are assumed.
    """
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
        # Those spins which should be |1>s get a positive weight which is a
        # penalty/reward for |0>/|1>, and those which should be |0>s get a
        # negative weight which is a reward/penalty for |0>/|1>.
        fixing_weights = WeightAccumulator(
            linear_weights={
                binary_variable_name: fixing_weight
                for binary_variable_name
                in variable_names[:number_of_ones]
            }
        )
        fixing_weights.add_linears({
            binary_variable_name: -fixing_weight
            for binary_variable_name
            in variable_names[number_of_ones:]
        })
        return fixing_weights

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
        FieldAtPoint to create a pair of dicts in the form for sample_ising.
        """
        # This is the case of a FieldAtPoint having correlations between its own
        # variables.
        domain_wall_weights = WeightTemplate(
            number_of_values_for_normal=number_of_variables,
            number_of_values_for_transpose=number_of_variables
        )

        # First, we set the weights to fix the ends so that there is a domain of
        # 1s from the first index and a domain of 0s ending at the last index.
        # Much like in weights_for_fixed_value above, the positive weight for
        # the first variable is a penalty/reward under spin or a penalty/nothing
        # under bit, and the negative weight for the last variable is a
        # reward/penalty under spin or a nothing/reward under bit.
        domain_wall_weights.linear_weights_for_normal[0] = end_weight
        domain_wall_weights.linear_weights_for_normal[-1] = -end_weight

        # Each pair of nearest neighbors gets weighted to favor having the same
        # values - which is either (-1)^2 or (+1)^2, so +1, while opposite
        # values multiply the weighting by (-1) * (+1) = -1. Therefore, a
        # negative weighting will penalize opposite spins with a positive
        # contribution to the objective function.
        for i in range(number_of_variables - 1):
            domain_wall_weights.quadratic_weights[i][i + 1] = -alignment_weight

        return domain_wall_weights

    def _set_alignment_weights_for_domain_wall(
            self,
            *,
            domain_wall_weights: WeightTemplate,
            number_of_variables: int,
            end_weight: float,
            alignment_weight: float
    ):
        # First, we set the weights to fix the ends so that there is a domain of
        # 1s from the first index and a domain of 0s ending at the last index.
        # The positive weight for the first spin is a penalty/reward, and the
        # negative weight for the last spin is a reward/penalty.
        domain_wall_weights.linear_weights_for_normal[0] = end_weight
        domain_wall_weights.linear_weights_for_normal[-1] = -end_weight

        # Each pair of nearest neighbors gets weighted to favor having the same
        # values - which is either (-1)^2 or (+1)^2, so +1, while opposite
        # values multiply the weighting by (-1) * (+1) = -1. Therefore, a
        # negative weighting will penalize opposite spins with a positive
        # contribution to the objective function.
        for i in range(number_of_variables - 1):
            domain_wall_weights.quadratic_weights[i][i + 1] = -alignment_weight
