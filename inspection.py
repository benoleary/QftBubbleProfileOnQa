import dwave.inspector
import minimization.sampling
import minimization.variable
from configuration.configuration import DiscreteConfiguration
from structure.bubble import BubbleProfile
from hamiltonian.field import FieldAtPoint
import hamiltonian.potential


def inspect_single_chain_for_single_field(sampler_name: str):
    test_field = FieldAtPoint(
        field_name="t",
        spatial_point_identifier="x0",
        number_of_values_for_field=7,
        field_step_in_GeV=1.0,
        offset_from_origin_in_GeV=0.0
    )
    end_weight = 10.0
    alignment_weight = 3.5
    spin_biases = test_field.weights_for_domain_wall(
            end_spin_weight=end_weight,
            spin_alignment_weight=alignment_weight
        )
    sampling_result = minimization.sampling.get_sample(
        spin_biases=spin_biases,
        message_for_Leap="Just a field as a single chain",
        number_of_shots=100,
        sampler_name=sampler_name
    )
    lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
    minimization.variable.print_bitstrings(
        "lowest energies for single field chain:",
        lowest_energy
    )
    dwave.inspector.show(sampling_result)
    print(sampling_result)

def flat_and_zigzag_from_kinetic_term(sampler_name: str):
    test_configuration = DiscreteConfiguration(
        first_field_name="f",
        number_of_spatial_steps=2,
        spatial_step_in_inverse_GeV=1.0,
        volume_exponent=0,
        first_field_step_in_GeV=1.0,
        first_field_offset_in_GeV=0.0,
        potential_in_quartic_GeV_per_field_step=[0.0 for _ in range(5)]
    )
    test_bubble_profile = BubbleProfile(test_configuration)

    penalizing_kinetic_result = minimization.sampling.get_sample(
        spin_biases=test_bubble_profile.spin_biases,
        message_for_Leap="Just kinetic weights expecting flat profile",
        number_of_shots=100,
        sampler_name=sampler_name
    )
    minimization.variable.print_bitstrings(
        "lowest energies for just kinetic term:",
        penalizing_kinetic_result.lowest(rtol=0.01, atol=0.1)
    )

    center_field = test_bubble_profile.fields_at_points[0].first_field
    intermediate_field = test_bubble_profile.fields_at_points[1].first_field
    outer_field = test_bubble_profile.fields_at_points[2].first_field
    radius_step = test_configuration.spatial_step_in_inverse_GeV
    kinetic_weights = hamiltonian.kinetic.weights_for_difference(
        at_smaller_radius=center_field,
        at_larger_radius=intermediate_field,
        radius_difference_in_inverse_GeV=radius_step
    )
    kinetic_weights.add(
        hamiltonian.kinetic.weights_for_difference(
            at_smaller_radius=intermediate_field,
            at_larger_radius=outer_field,
            radius_difference_in_inverse_GeV=radius_step
        )
    )
    weights_to_flip_kinetic = {
        k: -2.0 * v
        for k, v in kinetic_weights.quadratic_biases.items()
    }
    rewarding_kinetic = test_bubble_profile.spin_biases
    rewarding_kinetic.add_quadratics(weights_to_flip_kinetic)

    rewarding_kinetic_result = minimization.sampling.get_sample(
        spin_biases=rewarding_kinetic,
        message_for_Leap="Just kinetic weights expecting zig-zag profile",
        sampler_name=sampler_name
    )
    minimization.variable.print_bitstrings(
        "lowest energies for inverted kinetic term:",
        rewarding_kinetic_result.lowest(rtol=0.01, atol=0.1)
    )

    if minimization.sampling.is_online_sampler(sampler_name):
        dwave.inspector.show(penalizing_kinetic_result)
        print(penalizing_kinetic_result)

def low_resolution_single_field_with_linear_potential(sampler_name: str):
    test_configuration = DiscreteConfiguration(
        first_field_name="f",
        number_of_spatial_steps=4,
        spatial_step_in_inverse_GeV=1.0,
        volume_exponent=0,
        first_field_step_in_GeV=1.0,
        first_field_offset_in_GeV=0.0,
        potential_in_quartic_GeV_per_field_step=[0.6 * f for f in range(5)]
    )
    test_bubble_profile = BubbleProfile(test_configuration)

    full_result = minimization.sampling.get_sample(
        spin_biases=test_bubble_profile.spin_biases,
        message_for_Leap="Low resolution single field with linear potential",
        number_of_shots=100,
        sampler_name=sampler_name
    )
    minimization.variable.print_bitstrings(
        "lowest energies:",
        full_result.lowest(atol=100.0)
    )

    if sampler_name == "dwave":
        dwave.inspector.show(full_result)
        print(full_result)


if __name__ == "__main__":
    # inspect_single_chain_for_single_field("dwave")
    # flat_and_zigzag_from_kinetic_term("dwave")
    low_resolution_single_field_with_linear_potential("default")
