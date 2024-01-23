import dwave.inspector
import minimization.sampling
import basis.variable
from input.configuration import DiscreteConfiguration, FieldDefinition
from structure.bubble import BubbleProfile
from basis.field import FieldAtPoint, FieldDefinition
import dynamics.potential


def inspect_single_chain_for_single_field(sampler_name: str):
    test_field = FieldAtPoint(
        field_definition=FieldDefinition(
            field_name="t",
            number_of_values=7,
            lower_bound_in_GeV=0.0,
            upper_bound_in_GeV=6.0,
            true_vacuum_value_in_GeV=0.0,
            false_vacuum_value_in_GeV=6.0
        ),
        spatial_point_identifier="x0"
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
    basis.variable.print_bitstrings(
        "lowest energies for single field chain:",
        lowest_energy
    )

    if sampler_name == "dwave":
        dwave.inspector.show(sampling_result)
        print(sampling_result)


def flat_and_zigzag_from_kinetic_term(sampler_name: str):
    number_of_field_values = 5
    test_configuration = DiscreteConfiguration(
        number_of_spatial_steps=2,
        spatial_step_in_inverse_GeV=1.0,
        volume_exponent=0,
        first_field=FieldDefinition(
            field_name="f",
            number_of_values=number_of_field_values,
            lower_bound_in_GeV=0.0,
            upper_bound_in_GeV=4.0,
            true_vacuum_value_in_GeV=0.0,
            false_vacuum_value_in_GeV=4.0
        ),
        potential_in_quartic_GeV_per_field_step=[
            [0.0 for _ in range(number_of_field_values)]
        ]
    )
    test_bubble_profile = BubbleProfile(test_configuration)

    penalizing_kinetic_result = minimization.sampling.get_sample(
        spin_biases=test_bubble_profile.spin_biases,
        message_for_Leap="Just kinetic weights expecting flat profile",
        number_of_shots=100,
        sampler_name=sampler_name
    )
    basis.variable.print_bitstrings(
        "lowest energies for just kinetic term:",
        penalizing_kinetic_result.lowest(rtol=0.01, atol=0.1)
    )

    center_field = test_bubble_profile.fields_at_points[0].first_field
    intermediate_field = test_bubble_profile.fields_at_points[1].first_field
    outer_field = test_bubble_profile.fields_at_points[2].first_field
    radius_step = test_configuration.spatial_step_in_inverse_GeV
    kinetic_weights = dynamics.kinetic.weights_for_difference(
        at_smaller_radius=center_field,
        at_larger_radius=intermediate_field,
        radius_difference_in_inverse_GeV=radius_step
    )
    kinetic_weights.add(
        dynamics.kinetic.weights_for_difference(
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
    basis.variable.print_bitstrings(
        "lowest energies for inverted kinetic term:",
        rewarding_kinetic_result.lowest(rtol=0.01, atol=0.1)
    )

    if sampler_name == "dwave":
        dwave.inspector.show(penalizing_kinetic_result)
        print(penalizing_kinetic_result)


def low_resolution_single_field_with_linear_potential(sampler_name: str):
    number_of_field_values = 5
    test_configuration = DiscreteConfiguration(
        number_of_spatial_steps=4,
        spatial_step_in_inverse_GeV=1.0,
        volume_exponent=0,
        first_field=FieldDefinition(
            field_name="f",
            number_of_values=number_of_field_values,
            lower_bound_in_GeV=0.0,
            upper_bound_in_GeV=4.0,
            true_vacuum_value_in_GeV=0.0,
            false_vacuum_value_in_GeV=4.0
        ),
        potential_in_quartic_GeV_per_field_step=[
            [0.6 * f for f in range(number_of_field_values)]
        ]
    )
    test_bubble_profile = BubbleProfile(test_configuration)

    full_result = minimization.sampling.get_sample(
        spin_biases=test_bubble_profile.spin_biases,
        message_for_Leap="Low resolution single field with linear potential",
        number_of_shots=100,
        sampler_name=sampler_name
    )
    basis.variable.print_bitstrings(
        "lowest energies:",
        full_result.lowest(atol=100.0)
    )

    if sampler_name == "dwave":
        dwave.inspector.show(full_result)
        print(full_result)


if __name__ == "__main__":
    inspect_single_chain_for_single_field("dwave")
    # flat_and_zigzag_from_kinetic_term("default")
    # low_resolution_single_field_with_linear_potential("default")
