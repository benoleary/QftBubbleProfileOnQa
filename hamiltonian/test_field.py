import unittest
from dimod import ExactSolver
import minimization.variable
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
            "incorrect names for spin variables"
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

        actual_weights = test_field.domain_wall_weights(
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
        self.assertEqual(
            expected_linear_weights,
            actual_weights.linear_biases,
            "incorrect weights for linear biases"
        )
        self.assertEqual(
            expected_quadratic_weights,
            actual_weights.quadratic_biases,
            "incorrect weights for quadratic biases"
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
        spin_biases = test_field.domain_wall_weights(
                end_spin_weight=end_weight,
                spin_alignment_weight=alignment_weight
            )

        sampling_result = test_sampler.sample_ising(
            h=spin_biases.linear_biases,
            J=spin_biases.quadratic_biases
        )
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        actual_bitstrings_to_energies = {
            minimization.variable.as_bitstring(
                spin_variable_names=test_field.binary_variable_names,
                spin_mapping=s
            ): e
            for s, e in [(d.sample, d.energy) for d in lowest_energy.data()]
        }

        # We expect seven states, all with the same energy as a domain wall
        # between the first and second binary variables. The state 10000000
        # should have energy
        # -end_weight (from the first spin)
        # + alignment_weight (from the domain wall)
        # - 6 * alignment_weight (from the 6 pairs of aligned neighboring |0>s)
        # -end_weight (from the last spin)
        # = -2 * end_weight - 5 * alignment_weight
        expected_energy = -2.0 * end_weight - 5.0 * alignment_weight
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
        self.assertEqual(
            expected_bitstrings_to_energies,
            actual_bitstrings_to_energies,
            "incorrect weights for binary variables"
        )


if __name__ == "__main__":
    unittest.main()
