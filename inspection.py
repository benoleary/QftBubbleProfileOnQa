import dwave.inspector

from basis.field import FieldCollectionAtPoint, FieldDefinition
from dynamics.spin import SpinHamiltonian
from input.configuration import (
    QftModelConfiguration, SpatialLatticeConfiguration
)
from minimization.sampling import SampleProvider
from minimization.spin import SpinSamplerHandler
from structure.bubble import BubbleProfile
from structure.spin import SpinDomainWallWeighter


def inspect_single_chain_for_single_field(sampler_name: str):
    test_field_definition = FieldDefinition(
        field_name="t",
        number_of_values=7,
        lower_bound_in_GeV=0.0,
        upper_bound_in_GeV=6.0,
        true_vacuum_value_in_GeV=0.0,
        false_vacuum_value_in_GeV=6.0
    )
    test_fields_at_point = FieldCollectionAtPoint(
        spatial_point_identifier="x0",
        spatial_radius_in_inverse_GeV=1.0,
        first_field=test_field_definition
    )
    test_field_at_point=test_fields_at_point.first_field

    end_weight = 10.0
    alignment_weight = 3.5
    test_domain_wall_weighter = SpinDomainWallWeighter()
    test_weights = (
        test_domain_wall_weighter.weights_for_fixed_value(
            field_at_point=test_field_at_point,
            fixing_weight=end_weight,
            number_of_ones=(
                1 + test_field_definition.true_vacuum_value_in_steps
            )
        )
    )
    test_weights.add(
        test_domain_wall_weighter.weights_for_fixed_value(
            field_at_point=test_field_at_point,
            fixing_weight=end_weight,
            number_of_ones=(
                1 + test_field_definition.false_vacuum_value_in_steps
            )
        )
    )

    test_weights.add(
        test_domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=[test_fields_at_point],
            end_weight=end_weight,
            alignment_weight=alignment_weight
        )
    )

    test_sample_provider = SampleProvider(
        sampler_name=sampler_name,
        sampler_handler=SpinSamplerHandler(),
        message_for_Leap="Just a field as a single chain",
        number_of_shots=100
    )
    sampling_result = test_sample_provider.get_sample(test_weights)

    lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
    test_sample_provider.print_bitstrings(
        title_message="lowest energies for single field chain:",
        sample_set=lowest_energy
    )

    if sampler_name == "dwave":
        dwave.inspector.show(sampling_result)
        print(sampling_result)


def flat_and_zigzag_from_kinetic_term(sampler_name: str):
    number_of_field_values = 5
    QFT_model_configuration = QftModelConfiguration(
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
    spin_Hamiltonian = SpinHamiltonian(QFT_model_configuration)
    domain_wall_weighter = SpinDomainWallWeighter()
    radius_step = 1.0
    spatial_lattice_configuration = SpatialLatticeConfiguration(
        number_of_spatial_steps=2,
        spatial_step_in_inverse_GeV=radius_step,
        volume_exponent=0
    )

    bubble_profile = BubbleProfile(
        annealer_Hamiltonian=spin_Hamiltonian,
        domain_wall_weighter=domain_wall_weighter,
        spatial_lattice_configuration=spatial_lattice_configuration
    )

    test_sample_provider = SampleProvider(
        sampler_name=sampler_name,
        sampler_handler=SpinSamplerHandler(),
        message_for_Leap="Just kinetic weights expecting flat profile",
        number_of_shots=100
    )

    penalizing_kinetic_result = test_sample_provider.get_sample(
        bubble_profile.annealing_weights
    )
    test_sample_provider.print_bitstrings(
        title_message="lowest energies for just kinetic term:",
        sample_set=penalizing_kinetic_result.lowest(rtol=0.01, atol=0.1)
    )

    # This is a bit hacky, obbviously.
    kinetic_weights = bubble_profile._get_kinetic_weights(radius_step)
    weights_to_flip_kinetic = {
        k: -2.0 * v
        for k, v in kinetic_weights.quadratic_weights.items()
    }
    # Continuing the hackiness, the bubble profile's weights are now going to be
    # changed, but we do not need the original weights any more.
    rewarding_kinetic = bubble_profile.annealing_weights
    rewarding_kinetic.add_quadratics(weights_to_flip_kinetic)

    test_sample_provider.message_for_Leap = (
        "Just kinetic weights expecting zig-zag profile"
    )
    rewarding_kinetic_result = test_sample_provider.get_sample(
        rewarding_kinetic
    )
    test_sample_provider.print_bitstrings(
        title_message="lowest energies for inverted kinetic term:",
        sample_set=rewarding_kinetic_result.lowest(rtol=0.01, atol=0.1)
    )

    if sampler_name == "dwave":
        dwave.inspector.show(penalizing_kinetic_result)
        print(penalizing_kinetic_result)


def low_resolution_single_field_with_linear_potential(sampler_name: str):
    number_of_field_values = 5
    QFT_model_configuration = QftModelConfiguration(
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
    spin_Hamiltonian = SpinHamiltonian(QFT_model_configuration)
    domain_wall_weighter = SpinDomainWallWeighter()
    radius_step = 1.0
    spatial_lattice_configuration = SpatialLatticeConfiguration(
        number_of_spatial_steps=4,
        spatial_step_in_inverse_GeV=radius_step,
        volume_exponent=0
    )

    bubble_profile = BubbleProfile(
        annealer_Hamiltonian=spin_Hamiltonian,
        domain_wall_weighter=domain_wall_weighter,
        spatial_lattice_configuration=spatial_lattice_configuration
    )

    test_sample_provider = SampleProvider(
        sampler_name=sampler_name,
        sampler_handler=SpinSamplerHandler(),
        message_for_Leap="Low resolution single field with linear potential",
        number_of_shots=100
    )

    full_result = test_sample_provider.get_sample(
        bubble_profile.annealing_weights
    )
    test_sample_provider.print_bitstrings(
        title_message="lowest energies:",
        sample_set=full_result.lowest(atol=100.0)
    )

    if sampler_name == "dwave":
        dwave.inspector.show(full_result)
        print(full_result)


if __name__ == "__main__":
    inspect_single_chain_for_single_field("dwave")
    # flat_and_zigzag_from_kinetic_term("default")
    # low_resolution_single_field_with_linear_potential("default")
