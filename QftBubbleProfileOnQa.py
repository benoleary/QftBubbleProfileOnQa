# from dwave.system import DWaveSampler, EmbeddingComposite
# import dwave.inspector
from dimod import ExactSolver

# dwave_sampler = DWaveSampler()
# chosen_sampler = EmbeddingComposite(dwave_sampler)
chosen_sampler = ExactSolver()

# For the moment, we just use a simple example from the D-Wave Ocean
# documentation.
bqm_quadratic_part = \
    {
        ('x1', 'x2'): 1,
        ('x1', 'z'): -2,
        ('x2', 'z'): -2,
        ('z', 'z'): 3
    }
sampling_result = \
    chosen_sampler.sample_qubo(
        bqm_quadratic_part,
        # These kwargs are relevant only to the embedded sampler. The exact
        # solver ignores them (after complaining about being given unknown
        # kwargs).
        num_reads = 100,
        label = 'SDK Examples - AND Gate'
    )

# dwave.inspector.show(sample_set)
print(sampling_result)
