import unittest
from dimod import ExactSolver
from hamiltonian.field import FieldAtPoint

class TestFieldAtPoint(unittest.TestCase):
    def test_binary_variable_names_constructed_correctly(self):
        test_field = \
            FieldAtPoint(
                field_name = "t",
                spatial_point_identifier = "x0",
                number_of_active_binary_variables = 1
            )
        self.assertEqual(
            ["t_x0_0", "t_x0_1", "t_x0_2"],
            test_field.binary_variable_names,
            "incorrect names for binary variables"
        )

    def test_binary_variable_values_constructed_correctly(self):
        test_field = \
            FieldAtPoint(
                field_name = "t",
                spatial_point_identifier = "x",
                number_of_active_binary_variables = 4
            )
        expected_names_with_values = [
            ("t_x_0", 0.0),
            ("t_x_1", 0.25),
            ("t_x_2", 0.5),
            ("t_x_3", 0.75),
            ("t_x_4", 1.0)
        ]
        actual_names_with_values = \
            test_field.binary_variable_names_with_field_values
        print(f"actual_names_with_values = {actual_names_with_values}")
        self.assertEqual(
            len(expected_names_with_values),
            len(actual_names_with_values)
        )
        for i in range(len(expected_names_with_values)):
            self.assertAlmostEqual(
                expected_names_with_values[i],
                actual_names_with_values[i],
                f"incorrect name and value for binary variable {i}"
            )

    def test_domain_wall_weights_given_correctly(self):
        test_field = \
            FieldAtPoint(
                field_name = "t",
                spatial_point_identifier = "x",
                number_of_active_binary_variables = 2
            )
        end_weight = 10.0
        alignment_weight = 3.5

        actual_weights = \
            test_field.weights_for_ICDW(
                end_spin_weight = end_weight,
                spin_alignment_weight = alignment_weight
            ).weight_matrix

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
            actual_weights,
            "incorrect weights for binary variables"
        )

    def test_all_valid_strengths_for_only_domain_wall_conditions(self):
        test_sampler = ExactSolver()
        test_field = \
            FieldAtPoint(
                field_name = "t",
                spatial_point_identifier = "x0",
                number_of_active_binary_variables = 6
            )
        end_weight = 10.0
        alignment_weight = 3.5
        binary_quadratic_model = \
            test_field.weights_for_ICDW(
                end_spin_weight = end_weight,
                spin_alignment_weight = alignment_weight
            ).weight_matrix

        sampling_result = test_sampler.sample_qubo(binary_quadratic_model)
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        bitstrings_to_energies = {
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
        self.assertEqual(
            {b: expected_energy for b in expected_bitstrings},
            bitstrings_to_energies,
            "incorrect weights for binary variables"
        )


if __name__ == "__main__":
    unittest.main()
