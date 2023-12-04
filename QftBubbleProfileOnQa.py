from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector
from dimod import ExactSolver
from hamiltonian.field import FieldAtPoint
from hamiltonian.potential import SingleFieldPotential

dwave_sampler = DWaveSampler()
test_sampler = EmbeddingComposite(dwave_sampler)
# test_sampler = ExactSolver()
test_field = \
    FieldAtPoint(
        field_name = "t",
        spatial_point_identifier = "x0",
        number_of_active_binary_variables = 6
    )
end_weight = 10.0
alignment_weight = 3.5
weight_accumulator = \
    test_field.weights_for_ICDW(
        end_spin_weight = end_weight,
        spin_alignment_weight = alignment_weight
    )
# In this case as well, the linear term is irrelevant.
test_potential = SingleFieldPotential(lambda f : (2.0 * f) - 20.0)
weight_accumulator.add(
    test_potential.get_weights_for_potential(test_field).weight_matrix
)

sampling_result = \
    test_sampler.sample_qubo(
        weight_accumulator.weight_matrix,
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
