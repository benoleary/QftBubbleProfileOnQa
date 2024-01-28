import pytest

from basis.field import FieldAtPoint, FieldCollectionAtPoint, FieldDefinition
import basis.variable
from minimization.sampling import SampleProvider, SamplerHandler
from minimization.spin import SpinSamplerHandler
from structure.domain_wall import DomainWallWeighter
from structure.spin import SpinDomainWallWeighter


# TODO: bit versions
_spin_weighter = SpinDomainWallWeighter()
_spin_sampler_handler = SpinSamplerHandler()


class TestDomainWallWeighters():
    @pytest.mark.parametrize(
            "domain_wall_weighter, sampler_handler",
            [(_spin_weighter, _spin_sampler_handler)]
    )
    def test_all_valid_strengths_for_only_domain_wall_conditions(
        self,
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        test_field_definition = FieldDefinition(
            field_name="t",
            number_of_values=7,
            lower_bound_in_GeV=0.0,
            upper_bound_in_GeV=1.0,
            true_vacuum_value_in_GeV=0.0,
            false_vacuum_value_in_GeV=1.0
        )
        test_fields_at_point = FieldCollectionAtPoint(
                    spatial_point_identifier="x",
                    spatial_radius_in_inverse_GeV=1.0,
                    first_field=test_field_definition
                )
        end_weight = 10.0
        alignment_weight = 3.5
        annealing_weights = domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=[test_fields_at_point],
            end_weight=end_weight,
            alignment_weight=alignment_weight
        )

        test_sample_provider = SampleProvider(
            sampler_name="exact",
            sampler_handler=sampler_handler
        )
        sampling_result = test_sample_provider.get_sample(annealing_weights)
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        actual_bitstrings_to_energies = (
            basis.variable.bitstrings_to_energies(
                binary_variable_names=(
                    test_fields_at_point.first_field.binary_variable_names
                ),
                sample_set=lowest_energy
            )
        )

        # We expect seven states, all with the same energy, but this depends on
        # the exact weights and does not matter for the purposes of solving the
        # problem, so we just check that the expected states are there and all
        # have the same energy.
        expected_bitstrings = {
            "10000000",
            "11000000",
            "11100000",
            "11110000",
            "11111000",
            "11111100",
            "11111110"
        }
        assert (
            expected_bitstrings == actual_bitstrings_to_energies.keys()
        ), "incorrect states found"

        zero_energy = actual_bitstrings_to_energies["10000000"]
        # All the energies should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        deviating_states = [
            (k, v) for k, v in actual_bitstrings_to_energies.items()
            if v != zero_energy
        ]
        assert (
            [] == deviating_states
        ), "incorrect energies for found states"

    @pytest.mark.parametrize(
            "domain_wall_weighter, sampler_handler",
            [(_spin_weighter, _spin_sampler_handler)]
    )
    @pytest.mark.parametrize(
            "number_of_ones, expected_bitstring",
            [
                (1, "100000"),
                (2, "110000"),
                (3, "111000"),
                (4, "111100"),
                (5, "111110"),
                # We also test the negative input convention.
                (-1, "111110"),
                (-2, "111100"),
                (-3, "111000"),
                (-4, "110000"),
                (-5, "100000")
            ]
    )
    def test_fixing_value(
        self,
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler,
        number_of_ones: int,
        expected_bitstring: str
    ):
        test_field = FieldAtPoint(
            field_definition=FieldDefinition(
                field_name="t",
                number_of_values=5,
                lower_bound_in_GeV=0.0,
                upper_bound_in_GeV=1.0,
                true_vacuum_value_in_GeV=0.0,
                false_vacuum_value_in_GeV=1.0
            ),
            spatial_point_identifier="x"
        )
        fixing_weight = 11.0

        annealing_weights = domain_wall_weighter.weights_for_fixed_value(
            field_at_point=test_field,
            fixing_weight=fixing_weight,
            number_of_ones=number_of_ones
        )

        test_sample_provider = SampleProvider(
            sampler_name="exact",
            sampler_handler=sampler_handler
        )
        sampling_result = test_sample_provider.get_sample(annealing_weights)
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        actual_bitstrings_to_energies = (
            basis.variable.bitstrings_to_energies(
                binary_variable_names=test_field.binary_variable_names,
                sample_set=lowest_energy
            )
        )

        # Again, the exact energy depends on the exact weights and is ultimately
        # irrelevant.
        assert (
            {expected_bitstring} == actual_bitstrings_to_energies.keys()
        ), "incorrect state(s) found"
