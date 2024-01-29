from __future__ import annotations
from typing import Union

from dimod import Sampler, SampleSet

from minimization.weight import WeightAccumulator


class BitSamplerHandler:
    """
    This class implements the methods of the SamplerHandler Protocol, passing
    the weights as biases for bit variables to the sample_qubo method of the
    dimod.Sampler.
    """
    def perform_sampling(
            self,
            *,
            chosen_sampler: Sampler,
            weight_container: WeightAccumulator,
            additional_arguments: dict[str, Union[str, int]]
    ) -> SampleSet:
        appropriate_arguments = {
                "Q": weight_container.for_QUBO(),
                **additional_arguments
        }
        return chosen_sampler.sample_qubo(**appropriate_arguments)

    def is_in_one_state(self, variable_value: int) -> bool:
        return bool(variable_value)
