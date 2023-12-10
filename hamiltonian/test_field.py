import unittest
from dimod import ExactSolver
from hamiltonian.field import FieldAtPoint

class TestFieldAtPoint(unittest.TestCase):
    def test_binary_variable_names_constructed_correctly(self):
        test_field = FieldAtPoint(
                field_name="t",
                spatial_point_identifier="x0",
                number_of_values_for_field=2,
                field_step_in_GeV=1.0
            )
        self.assertEqual(
            ["t_x0_0", "t_x0_1", "t_x0_2"],
            test_field.binary_variable_names,
            "incorrect names for binary variables"
        )

    def test_domain_wall_weights_given_correctly(self):
        test_field = FieldAtPoint(
                field_name="t",
                spatial_point_identifier="x",
                number_of_values_for_field=3,
                field_step_in_GeV=1.0
            )
        end_weight = 10.0
        alignment_weight = 3.5

        actual_weights = test_field.weights_for_ICDW(
                end_spin_weight = end_weight,
                spin_alignment_weight = alignment_weight
            )

        # All the nearest-neighbor interactions should have weights that
        # penalize differing values. This is XOR but that translates to a
        # penalty for each binary variable with a correlation which cancels the
        # individual penalties if both are 1.
        # Also, the first (index 0) and last (index 4 - 1 = 3) binary variables
        # should have the weights which fix them to 1 for the first and 0 for
        # the last, so should be end_weight but negative for the first one and
        # positive for the last.
        # Hence, at index 0, negative end_weight is forcing 1, and positive
        # alignment_weight is forcing correlation with index 1.
        # At index 1, positive alignment_weight is forcing correlation with
        # index 0 and again with index 2.
        # At index 2, positive alignment_weight is forcing correlation with
        # index 1 and again with index 3.
        # Finally, at index 3, positive end_weight is forcing 0, and positive
        # alignment_weight is forcing correlation with index 2.
        # Then all the pairs have -2 * alignment_weight to cancel the sum if
        # they are both 1 (nothing needs to cancel if both are 0).
        # The linear terms are actually given by the diagonal elements.
        expected_weights = {
            ("t_x_0", "t_x_0"): -end_weight + alignment_weight,
            ("t_x_1", "t_x_1"): 2.0 * alignment_weight,
            ("t_x_2", "t_x_2"): 2.0 * alignment_weight,
            ("t_x_3", "t_x_3"): end_weight + alignment_weight,
            ("t_x_0", "t_x_1"): -2.0 * alignment_weight,
            ("t_x_1", "t_x_2"): -2.0 * alignment_weight,
            ("t_x_2", "t_x_3"): -2.0 * alignment_weight
        }

        self.assertEqual(
            expected_weights,
            actual_weights.weight_matrix,
            "incorrect weights for binary variables"
        )

    def test_all_valid_strengths_for_only_domain_wall_conditions(self):
        test_sampler = ExactSolver()
        test_field = FieldAtPoint(
                field_name="t",
                spatial_point_identifier="x0",
                number_of_values_for_field=7,
                field_step_in_GeV=1.0
            )
        end_weight = 10.0
        alignment_weight = 3.5
        binary_quadratic_model = test_field.weights_for_ICDW(
                end_spin_weight = end_weight,
                spin_alignment_weight = alignment_weight
            )

        sampling_result = test_sampler.sample_qubo(
            binary_quadratic_model.weight_matrix
        )
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        actual_bitstrings_to_energies = {
            "".join([f"{s[n]}" for n in test_field.binary_variable_names]): e
            for s, e in [(d.sample, d.energy) for d in lowest_energy.data()]
        }

        # We expect seven states, all with the same energy as a domain wall
        # between the first and second binary variables. In this case, all the
        # pairwise correlations are 0 and all the linear weights are multiplied
        # by 0 except for the first binary variable, so the expected energy is
        # simply the linear weight that it has.
        expected_energy = alignment_weight - end_weight
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
        self.assertEqual(
            expected_bitstrings_to_energies,
            actual_bitstrings_to_energies,
            "incorrect weights for binary variables"
        )


if __name__ == "__main__":
    unittest.main()
