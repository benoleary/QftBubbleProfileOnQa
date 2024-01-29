from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol, Union

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
    weighting. This class should also handle how to convert the sample into
    bitstrings.
    """
    def perform_sampling(
            self,
            *,
            chosen_sampler: Sampler,
            weight_container: WeightAccumulator,
            additional_arguments: dict[str, Union[str, int]]
    ) -> SampleSet:
        raise NotImplementedError("SamplerHandler is just a Protocol")

    def is_in_one_state(self, variable_value: int) -> bool:
        raise NotImplementedError("SamplerHandler is just a Protocol")


class SampleProvider:
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

    def get_lowest_from_set(self, sample_set: SampleSet) -> dict[str, float]:
        lowest_energy_sample, = next(
            sample_set.lowest().data(
                fields=["sample"],
                sorted_by="energy"
            )
        )
        return lowest_energy_sample

    def get_number_of_variables_in_one_state(
            self,
            *,
            variable_names: Iterable[str],
            sample_set: SampleSet
    ) -> int:
        # This counts every |1> in the given iterable up to the first |0>.
        total_count = 0
        for variable_name in variable_names:
            is_one = self.sampler_handler.is_in_one_state(
                sample_set.get(variable_name, 0)
            )
            if is_one:
                total_count += 1
            else:
                break
        return total_count

    def bitstrings_to_energies(
            self,
            *,
            binary_variable_names: Iterable[str],
            sample_set: SampleSet
    ) -> dict[str, float]:
        return {
            self._as_bitstring(
                variable_names=binary_variable_names,
                variable_value_mapping=s
            ): e
            for s, e in [(d.sample, d.energy) for d in sample_set.data()]
        }

    def print_bitstrings(self, *, title_message: str, sample_set: SampleSet):
        print(title_message)
        print(
            "[v for v in sample_set.variables] = \n",
            [v for v in sample_set.variables]
        )
        print(
            "bitstrings in above variable order to energies =\n",
            self.bitstrings_to_energies(
                binary_variable_names=sample_set.variables,
                sample_set=sample_set
            )
        )

    def _as_bitstring(
            self,
            *,
            variable_names: Iterable[str],
            variable_value_mapping: dict[str, int]
    ) -> str:
        return "".join(
            [
                self._bitchar_for(variable_value_mapping[n])
                for n in variable_names
            ]
        )

    def _bitchar_for(self, variable_value: int) -> str:
        return (
            "1" if self.sampler_handler.is_in_one_state(variable_value) else "0"
        )
