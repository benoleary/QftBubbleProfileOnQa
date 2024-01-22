import pytest
from structure.spin import SpinDomainWallWeighter
import minimization.sampling
import minimization.variable
from hamiltonian.field import FieldAtPoint, FieldDefinition


class TestSpinDomainWallWeighter():
    def test_weights_for_domain_wall_given_correctly(self):
        test_field = FieldAtPoint(
            field_definition=FieldDefinition(
                field_name="t",
                number_of_values=3,
                lower_bound_in_GeV=0.0,
                upper_bound_in_GeV=1.0,
                true_vacuum_value_in_GeV=0.0,
                false_vacuum_value_in_GeV=1.0
            ),
            spatial_point_identifier="x"
        )
        end_weight = 10.0
        alignment_weight = 3.5

        actual_weights = SpinDomainWallWeighter().weights_for_domain_wall(
            field_at_point=test_field,
            end_spin_weight=end_weight,
            spin_alignment_weight=alignment_weight
        )

        # All the spins are either |1> which multiplies its weight by -1 or |0>
        # which multiplies its weight by +1.
        # All the nearest-neighbor interactions should have weights that
        # penalize differing values, by a simple positve bias.
        # Also, the first (index 0) and last (index 4 - 1 = 3) binary variables
        # should have the weights which fix them to |1> for the first and |0>
        # for the last, so should be end_weight for the first one (to encourage
        # |1> bringing its -1) and negative for the last (to encourage |0>
        # bringing its +1).
        # Hence, at index 0, positive end_weight is forcing |1>, and positive
        # alignment_weight is forcing correlation with index 1.
        # At index 1, positive alignment_weight is forcing correlation with
        # index 0 and again with index 2.
        # At index 2, positive alignment_weight is forcing correlation with
        # index 1 and again with index 3.
        # Finally, at index 3, negative end_weight is forcing |0>, and positive
        # alignment_weight is forcing correlation with index 2.
        # The linear terms are actually given by the diagonal elements.
        expected_linear_weights = {
            "t_x_0": end_weight,
            "t_x_3": -end_weight
        }
        expected_quadratic_weights = {
            ("t_x_0", "t_x_1"): -alignment_weight,
            ("t_x_1", "t_x_2"): -alignment_weight,
            ("t_x_2", "t_x_3"): -alignment_weight
        }

        # All the weights should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        assert (
            expected_linear_weights == actual_weights.linear_biases
        ), "incorrect weights for linear biases"
        assert (
            expected_quadratic_weights == actual_weights.quadratic_biases
        ), "incorrect weights for quadratic biases"

    def test_all_valid_strengths_for_only_domain_wall_conditions(self):
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
        spin_biases = SpinDomainWallWeighter().weights_for_domain_wall(
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
            minimization.variable.bitstrings_to_energies(
                binary_variable_names=test_field.binary_variable_names,
                sample_set=lowest_energy
            )
        )

        # We expect seven states, all with the same energy as a domain wall
        # between the first and second binary variables. The state 10000000
        # should have energy
        # -end_weight (from the first spin)
        # + alignment_weight (from the domain wall)
        # - 6 * alignment_weight (from the 6 pairs of aligned neighboring |0>s)
        # -end_weight (from the last spin)
        # = -2 * end_weight - 5 * alignment_weight
        expected_energy = (-2.0 * end_weight) - (5.0 * alignment_weight)
        # The expected lowest energy states all start with 1 and end with 0, and
        # we expect the seven combinations where there are only 1s on the left
        # and 0s on the right.
        expected_bitstrings = [
            "10000000",
            "11000000",
            "11100000",
            "11110000",
            "11111000",
            "11111100",
            "11111110"
        ]
        expected_bitstrings_to_energies = {
            b: expected_energy for b in expected_bitstrings
        }
        # All the energies should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        assert (
            expected_bitstrings_to_energies == actual_bitstrings_to_energies
        ), "incorrect weights for binary variables"

    @pytest.mark.parametrize(
            "number_of_down_spins, expected_bitstring",
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
    def test_fixing_value(self, number_of_down_spins, expected_bitstring):
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
        # With 5 possible values for the field, there are 6 spins (the first and
        # last are fixed, and there are 5 values for 0 to 4 of the middle 4
        # spins being |1>). Each spin contributes its weight to the energy of
        # the sample (all negative by design).
        expected_energy = -6.0 * fixing_weight

        spin_biases = SpinDomainWallWeighter().weights_for_fixed_value(
            field_at_point=test_field,
            fixing_weight=fixing_weight,
            number_of_down_spins=number_of_down_spins
        )

        expected_linear_weights = {
            "t_x_0": fixing_weight,
            "t_x_1": (
                fixing_weight if (
                    number_of_down_spins >= 2
                    or 0 > number_of_down_spins >= -4
                )
                else -fixing_weight
            ),
            "t_x_2": (
                fixing_weight if (
                    number_of_down_spins >= 3
                    or 0 > number_of_down_spins >= -3
                )
                else -fixing_weight
            ),
            "t_x_3": (
                fixing_weight if (
                    number_of_down_spins >= 4
                    or 0 > number_of_down_spins >= -2
                )
                else -fixing_weight
            ),
            "t_x_4": (
                fixing_weight if (
                    number_of_down_spins >= 5
                    or number_of_down_spins == -1
                )
                else -fixing_weight
            ),
            "t_x_5": -fixing_weight
        }

        # All the weights should be exactly representable in binary so
        # we can make floating-point number comparisons without needing
        # a tolerance.
        assert (
            expected_linear_weights == spin_biases.linear_biases
        ), "incorrect weights for linear biases"
        assert (
            {} == spin_biases.quadratic_biases
        ), "incorrect weights for quadratic biases"

        sampling_result = minimization.sampling.get_sample(
            spin_biases=spin_biases,
            sampler_name="exact"
        )
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        actual_bitstrings_to_energies = (
            minimization.variable.bitstrings_to_energies(
                binary_variable_names=test_field.binary_variable_names,
                sample_set=lowest_energy
            )
        )

        # All the energies should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        expected_bitstrings_to_energies = {expected_bitstring: expected_energy}
        assert (
            expected_bitstrings_to_energies == actual_bitstrings_to_energies
        ), "incorrect weights for binary variables"
