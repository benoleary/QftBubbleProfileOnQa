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
    return 0.0

number_of_potential_values = 5
discretized_potential = [
    single_field_potential(f) for f in range(number_of_potential_values)
]
test_configuration = DiscreteConfiguration(
    first_field_name="f",
    number_of_spatial_steps=2,
    spatial_step_in_inverse_GeV=1.0,
    field_step_in_GeV=1.0,
    potential_in_quartic_GeV_per_field_step=discretized_potential
)
test_bubble_profile = BubbleProfile(test_configuration)
test_field = test_bubble_profile.fields_at_points[0].first_field

penalizing_kinetic = test_bubble_profile.spin_biases

penalizing_kinetic_result = test_sampler.sample_ising(
    h=penalizing_kinetic.linear_biases,
    J=penalizing_kinetic.quadratic_biases,
    # These kwargs are relevant only to the embedded sampler. The exact solver
    # ignores them (after complaining about being given unknown kwargs).
    num_reads=100,
    label='SDK Examples - AND Gate'
)

kinetic_weights = hamiltonian.kinetic.weights_for_difference(
    at_smaller_radius=test_bubble_profile.fields_at_points[0].first_field,
    at_larger_radius=test_bubble_profile.fields_at_points[1].first_field,
    radius_difference_in_inverse_GeV=(
        test_configuration.spatial_step_in_inverse_GeV
    )
)
kinetic_weights.add(
    hamiltonian.kinetic.weights_for_difference(
        at_smaller_radius=test_bubble_profile.fields_at_points[1].first_field,
        at_larger_radius=test_bubble_profile.fields_at_points[2].first_field,
        radius_difference_in_inverse_GeV=(
            test_configuration.spatial_step_in_inverse_GeV
        )
    )
)
weights_to_flip_kinetic = {
    k: -2.0 * v
    for k, v in kinetic_weights.quadratic_biases.items()
}
rewarding_kinetic = test_bubble_profile.spin_biases
rewarding_kinetic.add_quadratics(weights_to_flip_kinetic)

rewarding_kinetic_result = test_sampler.sample_ising(
    h=rewarding_kinetic.linear_biases,
    J=rewarding_kinetic.quadratic_biases,
    # These kwargs are relevant only to the embedded sampler. The exact solver
    # ignores them (after complaining about being given unknown kwargs).
    num_reads=100,
    label='SDK Examples - AND Gate'
)

sample_lowest = penalizing_kinetic_result.lowest(rtol=0.01, atol=0.1)
print("penalizing kinetic:")
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
# print(penalizing_kinetic_result)

sample_lowest = rewarding_kinetic_result.lowest(rtol=0.01, atol=0.1)
print("rewarding kinetic:")
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
# print(rewarding_kinetic_result)
