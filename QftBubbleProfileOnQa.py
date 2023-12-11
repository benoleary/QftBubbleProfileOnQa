from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector
from dimod import ExactSolver
import minimization.variable
from configuration.configuration import DiscreteConfiguration
from structure.bubble import BubbleProfile
import hamiltonian.potential

# dwave_sampler = DWaveSampler()
# test_sampler = EmbeddingComposite(dwave_sampler)
test_sampler = ExactSolver()

def single_field_potential(field_value: int):
    return (2.5 * (field_value - 3) * (field_value - 3)) + 10.0

number_of_potential_values = 5
discretized_potential = [
    single_field_potential(f) for f in range(number_of_potential_values)
]
test_configuration = DiscreteConfiguration(
    first_field_name="f",
    number_of_spatial_steps=1,
    spatial_step_in_inverse_GeV=1.0,
    field_step_in_GeV=1.0,
    potential_in_quartic_GeV_per_field_step=discretized_potential
)
test_bubble_profile = BubbleProfile(test_configuration)
test_field = test_bubble_profile.fields_at_points[0].first_field

potential_weights = hamiltonian.potential.weights_for(
    test_configuration,
    test_field
)

end_weight = 10.0
alignment_weight = 3.5
weight_accumulator = test_field.domain_wall_weights(
        end_spin_weight=end_weight,
        spin_alignment_weight=alignment_weight
    )
weight_accumulator.add(potential_weights)

sampling_result = test_sampler.sample_ising(
    h=weight_accumulator.linear_biases,
    J=weight_accumulator.quadratic_biases,
    # These kwargs are relevant only to the embedded sampler. The exact solver
    # ignores them (after complaining about being given unknown kwargs).
    num_reads=100,
    label='SDK Examples - AND Gate'
)

sample_lowest = sampling_result.lowest(rtol=0.01, atol=0.1)
print(
    "[v for v in sample_lowest.variables] = \n",
    [v for v in sample_lowest.variables]
)
print(
    "bitstrings in above variable order to energies for lowest results = \n",
    {
            minimization.variable.as_bitstring(
                spin_variable_names=sample_lowest.variables,
                spin_mapping=s
            ): e
            for s, e in [(d.sample, d.energy) for d in sample_lowest.data()]
    }
)
# dwave.inspector.show(sample_set)
print(sampling_result)
