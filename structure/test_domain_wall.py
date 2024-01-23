import pytest

from basis.field import FieldAtPoint, FieldDefinition
import basis.variable
import minimization.sampling
from structure.domain_wall import DomainWallWeighter
from structure.spin import SpinDomainWallWeighter


# TODO: bit version
_spin_weighter = SpinDomainWallWeighter()

class TestDomainWallWeighters():
    @pytest.mark.parametrize(
            "domain_wall_weighter",
            [(_spin_weighter,)]
    )
    def test_all_valid_strengths_for_only_domain_wall_conditions(
        self,
        domain_wall_weighter: DomainWallWeighter
    ):
        # TODO: fix this
        test_field = FieldAtPoint(
            field_definition=FieldDefinition(
                field_name="t",
                number_of_values=7,
                lower_bound_in_GeV=0.0,
                upper_bound_in_GeV=1.0,
                true_vacuum_value_in_GeV=0.0,
                false_vacuum_value_in_GeV=1.0
            ),
            spatial_point_identifier="x0"
        )
        end_weight = 10.0
        alignment_weight = 3.5
        spin_biases = domain_wall_weighter.weights_for_domain_walls(
            field_at_point=test_field,
            end_spin_weight=end_weight,
            spin_alignment_weight=alignment_weight
        )

        sampling_result = minimization.sampling.get_sample(
            spin_biases=spin_biases,
            sampler_name="exact"
        )
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        actual_bitstrings_to_energies = (
            basis.variable.bitstrings_to_energies(
                binary_variable_names=test_field.binary_variable_names,
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
            "domain_wall_weighter",
            [(_spin_weighter,)]
    )
    @pytest.mark.parametrize(
            "domain_wall_weighter", "number_of_ones, expected_bitstring",
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
        domain_wall_weighter,
        number_of_ones,
        expected_bitstring
    ):
        # TODO: fix this
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

        spin_biases = SpinDomainWallWeighter().weights_for_fixed_value(
            field_at_point=test_field,
            fixing_weight=fixing_weight,
            number_of_ones=number_of_ones
        )

        sampling_result = minimization.sampling.get_sample(
            spin_biases=spin_biases,
            sampler_name="exact"
        )
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
