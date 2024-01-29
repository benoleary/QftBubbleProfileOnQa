from __future__ import annotations
from collections.abc import Sequence

from minimization.weight import WeightAccumulator, WeightTemplate
from structure.domain_wall import TemplateDomainWallWeighter


class BitDomainWallWeighter(TemplateDomainWallWeighter):
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
        # Those bits which should be |1>s get a negative weight which is a
        # nothing/penalty for |0>/|1>, and those which should be |0>s get a
        # positive weight which is a penalty/nothing for |0>/|1>.
        fixing_weights = WeightAccumulator(
            linear_weights={
                binary_variable_name: -fixing_weight
                for binary_variable_name
                in variable_names[:number_of_ones]
            }
        )
        fixing_weights.add_linears({
            binary_variable_name: fixing_weight
            for binary_variable_name
            in variable_names[number_of_ones:]
        })
        return fixing_weights

    def _set_weights_for_domain_wall(
            self,
            *,
            domain_wall_weights: WeightTemplate,
            number_of_variables: int,
            end_weight: float,
            alignment_weight: float
    ):
        # First, we set the weights to fix the ends so that there is a domain of
        # 1s from the first index and a domain of 0s ending at the last index.
        # The negative weight for the first bit is a nothing/reward, and the
        # positive weight for the last bit is a penalty/nothing.
        domain_wall_weights.linear_weights_for_normal[0] = -end_weight
        domain_wall_weights.linear_weights_for_normal[-1] = end_weight

        # In the bit variable case, 00 automatically has zero weight, so we go
        # with zero weight for the aligned cases and a penalty weight of
        # alignment_weight for the non-aligned cases. This leads to weighting
        # the linear terms then cancelling the penalty with additional negative
        # weighting for 11.
        # Since the first variable is "fixed" to 1 anyway, 100..0 has zero
        # penalty from alignment. If we add alignment_weight for each variable
        # except the first and the last while subtracting alignment_weight for
        # each pair of nearest neighbors (including the first+second pair and
        # send-last+last pair), then going from 100..0 to 110..0 stays at zero
        # penalty, and generally from 1..100..0 to 1..110..0 while adding in
        # another domain of X 1s in the middle of 0..0 brings a penalty of
        # (X * alignment_weight) compensated by ((X - 1) * alignment_weight)
        # from the (X - 1) neighbor pairs, so bringing a penalty of
        # alignment_weight. Another domain of 0s in the middle of 1..1 brings
        # the same penalty.
        # Hence we just need to give linear penalty weights to the second
        # variable to the second-last, and a cancelling reward weight to each
        # with its lower-index neighbor. The last variable is fixed anyway and
        # the second-last+last alignment will never be 11 so its weight is
        # irrelevant.
        for i in range(1, (number_of_variables - 1)):
            domain_wall_weights.linear_weights_for_normal[i] = alignment_weight
            domain_wall_weights.quadratic_weights[i - 1][i] = -alignment_weight
