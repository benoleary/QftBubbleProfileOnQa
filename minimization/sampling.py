from typing import Callable, Dict
from dwave.system import DWaveSampler, EmbeddingComposite
from hybrid.reference.kerberos import KerberosSampler
from dimod import ExactSolver, Sampler, SampleSet, SimulatedAnnealingSampler
from minimization.weight import WeightAccumulator


def get_sampler(sampler_name: str) -> Sampler:
    if sampler_name == "dwave":
        return EmbeddingComposite(DWaveSampler())
    if sampler_name == "kerberos":
        return KerberosSampler()
    if sampler_name == "exact":
        return ExactSolver()
    return SimulatedAnnealingSampler()


# TODO: replace with using SamplePovider.get_sample
def get_sample(
        *,
        spin_biases: WeightAccumulator,
        message_for_Leap: str = None,
        number_of_shots: int = 100,
        sampler_name: str
) -> SampleSet:
    chosen_sampler = get_sampler(sampler_name)
    appropriate_arguments = {
            "h": spin_biases.linear_biases,
            "J": spin_biases.quadratic_biases
    }
    if message_for_Leap and (sampler_name == "dwave"):
        appropriate_arguments["num_reads"] = number_of_shots
        appropriate_arguments["label"] = message_for_Leap
    if "kerberos" == sampler_name:
        appropriate_arguments["max_iter"] = 10
    return chosen_sampler.sample_ising(**appropriate_arguments)

def get_lowest_sample_from_set(sample_set: SampleSet) -> Dict[str, float]:
    lowest_energy_sample, = next(
        sample_set.lowest().data(
            fields=["sample"],
            sorted_by="energy"
        )
    )
    return lowest_energy_sample


# TODO: define functions to give something for perform_sampling.

class SamplePovider:
    """
    This class encapsulates both which dimod.Sampler is used and whether the
    weights are to be interpreted as bit weights or spin weights.
    """
    def __init__(
            self,
            *,
            message_for_Leap: str = None,
            number_of_shots: int = 100,
            sampler_name: str,
            perform_sampling: Callable[[Sampler, WeightAccumulator], SampleSet]
    ):
        self.message_for_Leap = message_for_Leap
        self.number_of_shots = number_of_shots
        self.chosen_sampler = get_sampler(sampler_name)
        self.perform_sampling = perform_sampling

    def get_sample(self, weight_container: WeightAccumulator) -> SampleSet:
        # TODO: message_for_Leap and other kwargs
        return self.perform_sampling(self.chosen_sampler, weight_container)
