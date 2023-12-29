from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import Sampler, SampleSet
from minimization.weight import BiasAccumulator


# This uses the presence or absence of a message for Leap to choose whether or
# not to sample with the Cloud service.
def get_sample(
        *,
        spin_biases: BiasAccumulator,
        message_for_Leap: str = None,
        number_of_shots: int = 100,
        local_sampler: Sampler = None
    ) -> SampleSet:
    if message_for_Leap:
        return EmbeddingComposite(DWaveSampler()).sample_ising(
            h=spin_biases.linear_biases,
            J=spin_biases.quadratic_biases,
            num_reads=number_of_shots,
            label=message_for_Leap
        )
    if local_sampler:
        return local_sampler.sample_ising(
            h=spin_biases.linear_biases,
            J=spin_biases.quadratic_biases
        )

    raise ValueError(
        "No message for Leap and no local sampler provided, so cannot run."
    )
