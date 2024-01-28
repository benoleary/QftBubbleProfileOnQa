from typing import Dict, Protocol, Union
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


class SamplerHandler(Protocol):
    """
    This class defines the method by which a given dimod.Sampler is used to get
    a dimod.SampleSet, by choosing the Sampler method appropriate to the kind of
    weighting.
    """
    def perform_sampling(
            self,
            *,
            chosen_sampler: Sampler,
            weight_container: WeightAccumulator,
            additional_arguments: Dict[str, Union[str, int]]
    ) -> SampleSet:
        raise NotImplementedError("SamplerHandler is just a Protocol")


class SamplePovider:
    """
    This class encapsulates both which dimod.Sampler is used and, through
    sampler_handler, how it is used.
    """
    def __init__(
            self,
            *,
            sampler_name: str,
            sampler_handler: SamplerHandler,
            message_for_Leap: str = None,
            number_of_shots: int = 100
    ):
        self.sampler_name = sampler_name
        self.sampler_handler = sampler_handler
        self.message_for_Leap = message_for_Leap
        self.number_of_shots = number_of_shots
        self.chosen_sampler = get_sampler(sampler_name)

    def get_sample(self, weight_container: WeightAccumulator) -> SampleSet:
        appropriate_arguments = {}
        if self.sampler_name == "dwave":
            appropriate_arguments["num_reads"] = self.number_of_shots
            if self.message_for_Leap:
                appropriate_arguments["label"] = self.message_for_Leap
        if "kerberos" == self.sampler_name:
            appropriate_arguments["max_iter"] = 10

        return self.sampler_handler.perform_sampling(
            chosen_sampler=self.chosen_sampler,
            weight_container=weight_container,
            additional_arguments=appropriate_arguments
        )

    def get_lowest_from_set(self, sample_set: SampleSet) -> Dict[str, float]:
        lowest_energy_sample, = next(
            sample_set.lowest().data(
                fields=["sample"],
                sorted_by="energy"
            )
        )
        return lowest_energy_sample
