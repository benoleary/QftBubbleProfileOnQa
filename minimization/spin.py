from typing import Dict, Union
from dimod import Sampler, SampleSet

from minimization.weight import WeightAccumulator


class SpinSamplerHandler:
    """
    This class implements the methods of the SamplerHandler Protocol, passing
    the weights as biases for spin variables to the sample_ising method of the
    dimod.Sampler.
    """
    def perform_sampling(
            self,
            *,
            chosen_sampler: Sampler,
            weight_container: WeightAccumulator,
            additional_arguments: Dict[str, Union[str, int]]
    ) -> SampleSet:
        appropriate_arguments = {
                "h": weight_container.linear_biases,
                "J": weight_container.quadratic_biases,
                **additional_arguments
        }
        return chosen_sampler.sample_ising(**appropriate_arguments)
