# from dwave.system import DWaveSampler, EmbeddingComposite
# import dwave.inspector
from dimod import ExactSolver
from hamiltonian.field import Field

# dwave_sampler = DWaveSampler()
# chosen_sampler = EmbeddingComposite(dwave_sampler)
chosen_sampler = ExactSolver()

# For the moment, we just use a simple example from the D-Wave Ocean
# documentation.
single_field = \
    Field(
        field_name = "t",
        spatial_point_identifier = "x0",
        number_of_binary_variables = 4
    )
binary_quadratic_model = \
    single_field.get_weights_for_ICDW(
        end_spin_weight = 10.0,
        spin_alignment_weight = 3.5
    )
sampling_result = \
    chosen_sampler.sample_qubo(
        binary_quadratic_model,
        # These kwargs are relevant only to the embedded sampler. The exact
        # solver ignores them (after complaining about being given unknown
        # kwargs).
        num_reads = 100,
        label = 'SDK Examples - AND Gate'
    )

sample_lowest = sampling_result.lowest(rtol=0.01, atol=0.1)
print(
    "[v for v in sample_lowest.variables] = \n",
    [v for v in sample_lowest.variables]
)
print(
    "bitstrings in above variable order to energies for lowest results = \n",
    {
        "".join([f"{x[v]}" for v in sample_lowest.variables]): y
        for x, y in [(d.sample, d.energy) for d in sample_lowest.data()]
    }
)
# dwave.inspector.show(sample_set)
print(sampling_result)
