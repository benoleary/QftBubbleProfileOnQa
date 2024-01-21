from typing import Dict
from dwave.system import DWaveSampler, EmbeddingComposite
from hybrid.reference.kerberos import KerberosSampler
from dimod import ExactSolver, Sampler, SampleSet, SimulatedAnnealingSampler
from minimization.weight import BiasAccumulator


def get_sampler(sampler_name: str) -> Sampler:
    if sampler_name == "dwave":
        return EmbeddingComposite(DWaveSampler())
    if sampler_name == "kerberos":
        return KerberosSampler()
    if sampler_name == "exact":
        return ExactSolver()
    return SimulatedAnnealingSampler()


# TODO: take sampling function (from configuration? from bubble?) rather than always "sample_ising"
# (might be able to simplify using kwargs idiomatically)
def get_sample(
        *,
        spin_biases: BiasAccumulator,
        message_for_Leap: str = None,
        number_of_shots: int = 100,
        sampler_name: str
) -> SampleSet:
    chosen_sampler = get_sampler(sampler_name)
    if message_for_Leap and (sampler_name == "dwave"):
        return chosen_sampler.sample_ising(
            h=spin_biases.linear_biases,
            J=spin_biases.quadratic_biases,
            num_reads=number_of_shots,
            label=message_for_Leap
        )
    if "kerberos" == sampler_name:
        return chosen_sampler.sample_ising(
            h=spin_biases.linear_biases,
            J=spin_biases.quadratic_biases,
            max_iter=10
        )
    return chosen_sampler.sample_ising(
        h=spin_biases.linear_biases,
        J=spin_biases.quadratic_biases
    )

def get_lowest_sample_from_set(sample_set: SampleSet) -> Dict[str, float]:
    lowest_energy_sample, = next(
        sample_set.lowest().data(
            fields=["sample"],
            sorted_by="energy"
        )
    )
    return lowest_energy_sample
