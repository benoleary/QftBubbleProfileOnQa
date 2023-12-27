from typing import Tuple
import unittest
from dimod import ExactSolver
import minimization.variable
from hamiltonian.field import FieldAtPoint
import hamiltonian.kinetic

class TestKinetic(unittest.TestCase):
    def setUp(self):
        self.field_name = "T"
        self.field_step_in_GeV = 7.0
        self.radius_step_in_inverse_GeV = 0.25
        # The weight should be (1/8) * 7^2 * 4^2 = (49 * 2) = 98
        self.absolute_expected_weight = 98.0

    def test_kinetic_weights_for_low_resolution_fields(self):
        """
        This tests that the correct weights are generated using an almost
        minimal number of spins for the fields.
        """
        # Each field should have 3 valid bitstrings: |1000>, |1100>, and |1110>.
        lower_radius_field, upper_radius_field = self._set_up_fields(3)
        actual_weights = hamiltonian.kinetic.weights_for_difference(
            at_smaller_radius=lower_radius_field,
            at_larger_radius=upper_radius_field,
            radius_difference_in_inverse_GeV=self.radius_step_in_inverse_GeV
        )

        # Only the spins which can vary should get weights, so there should be
        # no appearaces of T_l_0, T_l_3, T_r_0, or T_r_3.
        expected_quadratic_weights = {
            # The lower field with itself should be all positive
            ("T_l_1", "T_l_1"): self.absolute_expected_weight,
            ("T_l_1", "T_l_2"): self.absolute_expected_weight,
            ("T_l_2", "T_l_1"): self.absolute_expected_weight,
            ("T_l_2", "T_l_2"): self.absolute_expected_weight,
            # The lower field with the upper field should be all negative
            ("T_l_1", "T_u_1"): -self.absolute_expected_weight,
            ("T_l_1", "T_u_2"): -self.absolute_expected_weight,
            ("T_l_2", "T_u_1"): -self.absolute_expected_weight,
            ("T_l_2", "T_u_2"): -self.absolute_expected_weight,
            # The upper field with itself should be all positive
            ("T_u_1", "T_u_1"): self.absolute_expected_weight,
            ("T_u_1", "T_u_2"): self.absolute_expected_weight,
            ("T_u_2", "T_u_1"): self.absolute_expected_weight,
            ("T_u_2", "T_u_2"): self.absolute_expected_weight,
            # The upper field with the lower field should be all negative
            ("T_u_1", "T_l_1"): -self.absolute_expected_weight,
            ("T_u_1", "T_l_2"): -self.absolute_expected_weight,
            ("T_u_2", "T_l_1"): -self.absolute_expected_weight,
            ("T_u_2", "T_l_2"): -self.absolute_expected_weight
        }

        # All the weights should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        self.assertEqual(
            {},
            actual_weights.linear_biases,
            "incorrect weights for linear biases"
        )
        self.assertEqual(
            len(expected_quadratic_weights.keys()),
            len(actual_weights.quadratic_biases.keys()),
            "incorrect number of keys for quadratic biases"
        )
        self.assertEqual(
            expected_quadratic_weights.keys(),
            actual_weights.quadratic_biases.keys(),
            "incorrect keys for quadratic biases"
        )
        self.assertEqual(
            expected_quadratic_weights,
            actual_weights.quadratic_biases,
            "incorrect weights for quadratic biases"
        )

    def test_kinetic_weights_for_sightly_higher_resolution_fields(self):
        """
        This tests that the correct weights are generated using a small but not
        minimal number of spins for the fields
        """
        # Each field should have 4 valid bitstrings:
        # |10000>, |11000>, |11100>, and |11110>.
        lower_radius_field, upper_radius_field = self._set_up_fields(4)
        actual_weights = hamiltonian.kinetic.weights_for_difference(
            at_smaller_radius=lower_radius_field,
            at_larger_radius=upper_radius_field,
            radius_difference_in_inverse_GeV=self.radius_step_in_inverse_GeV
        )

        # Only the spins which can vary should get weights, so there should be
        # no appearaces of T_l_0, T_l_4, T_r_0, or T_r_4.
        expected_quadratic_weights = {
            # The lower field with itself should be all positive
            ("T_l_1", "T_l_1"): self.absolute_expected_weight,
            ("T_l_1", "T_l_2"): self.absolute_expected_weight,
            ("T_l_1", "T_l_3"): self.absolute_expected_weight,
            ("T_l_2", "T_l_1"): self.absolute_expected_weight,
            ("T_l_2", "T_l_2"): self.absolute_expected_weight,
            ("T_l_2", "T_l_3"): self.absolute_expected_weight,
            ("T_l_3", "T_l_1"): self.absolute_expected_weight,
            ("T_l_3", "T_l_2"): self.absolute_expected_weight,
            ("T_l_3", "T_l_3"): self.absolute_expected_weight,
            # The lower field with the upper field should be all negative
            ("T_l_1", "T_u_1"): -self.absolute_expected_weight,
            ("T_l_1", "T_u_2"): -self.absolute_expected_weight,
            ("T_l_1", "T_u_3"): -self.absolute_expected_weight,
            ("T_l_2", "T_u_1"): -self.absolute_expected_weight,
            ("T_l_2", "T_u_2"): -self.absolute_expected_weight,
            ("T_l_2", "T_u_3"): -self.absolute_expected_weight,
            ("T_l_3", "T_u_1"): -self.absolute_expected_weight,
            ("T_l_3", "T_u_2"): -self.absolute_expected_weight,
            ("T_l_3", "T_u_3"): -self.absolute_expected_weight,
            # The upper field with itself should be all positive
            ("T_u_1", "T_u_1"): self.absolute_expected_weight,
            ("T_u_1", "T_u_2"): self.absolute_expected_weight,
            ("T_u_1", "T_u_3"): self.absolute_expected_weight,
            ("T_u_2", "T_u_1"): self.absolute_expected_weight,
            ("T_u_2", "T_u_2"): self.absolute_expected_weight,
            ("T_u_2", "T_u_3"): self.absolute_expected_weight,
            ("T_u_3", "T_u_1"): self.absolute_expected_weight,
            ("T_u_3", "T_u_2"): self.absolute_expected_weight,
            ("T_u_3", "T_u_3"): self.absolute_expected_weight,
            # The upper field with the lower field should be all negative
            ("T_u_1", "T_l_1"): -self.absolute_expected_weight,
            ("T_u_1", "T_l_2"): -self.absolute_expected_weight,
            ("T_u_1", "T_l_3"): -self.absolute_expected_weight,
            ("T_u_2", "T_l_1"): -self.absolute_expected_weight,
            ("T_u_2", "T_l_2"): -self.absolute_expected_weight,
            ("T_u_2", "T_l_3"): -self.absolute_expected_weight,
            ("T_u_3", "T_l_1"): -self.absolute_expected_weight,
            ("T_u_3", "T_l_2"): -self.absolute_expected_weight,
            ("T_u_3", "T_l_3"): -self.absolute_expected_weight
        }

        # All the weights should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        self.assertEqual(
            {},
            actual_weights.linear_biases,
            "incorrect weights for linear biases"
        )
        self.assertEqual(
            len(expected_quadratic_weights.keys()),
            len(actual_weights.quadratic_biases.keys()),
            "incorrect number of keys for quadratic biases"
        )
        self.assertEqual(
            expected_quadratic_weights.keys(),
            actual_weights.quadratic_biases.keys(),
            "incorrect keys for quadratic biases"
        )
        self.assertEqual(
            expected_quadratic_weights,
            actual_weights.quadratic_biases,
            "incorrect weights for quadratic biases"
        )

    def test_all_valid_bitstrings_present_with_correct_energies(self):
        """
        This tests that the exact solver finds the correct energies for each
        valid pair of ICDW configurations of a field at neighboring radius
        values.
        """
        lower_radius_field, upper_radius_field = self._set_up_fields(4)
        # The weights from differences should go up to 0.5 * (4 * 7.0)^2 * 3^2
        # = 9 * 384 = 3456
        # so the ICDW weights need to be significantly larger than that.
        end_weight = 10000.0
        alignment_weight = 5000.0
        kinetic_weights = hamiltonian.kinetic.weights_for_difference(
            at_smaller_radius=lower_radius_field,
            at_larger_radius=upper_radius_field,
            radius_difference_in_inverse_GeV=self.radius_step_in_inverse_GeV
        )
        kinetic_weights.add(
            lower_radius_field.domain_wall_weights(
                end_spin_weight=end_weight,
                spin_alignment_weight=alignment_weight
            )
        )
        kinetic_weights.add(
            upper_radius_field.domain_wall_weights(
                end_spin_weight=end_weight,
                spin_alignment_weight=alignment_weight
            )
        )

        test_sampler = ExactSolver()
        sampling_result = test_sampler.sample_ising(
            h=kinetic_weights.linear_biases,
            J=kinetic_weights.quadratic_biases
        )

        # We grab all the results under the energy which should only happen if
        # the spins violate the conditions of the Ising-chain domain wall.
        samples_under_penalty_weight = sampling_result.lowest(
            atol=(0.75 * alignment_weight)
        )
        actual_bitstrings_to_energies = {
            minimization.variable.as_bitstring(
                spin_variable_names=(
                    lower_radius_field.binary_variable_names
                    + upper_radius_field.binary_variable_names
                ),
                spin_mapping=s
            ): e
            for s, e in [
                (d.sample, d.energy)
                for d in samples_under_penalty_weight.data()
            ]
        }

        # As in test_field.py, the base energy from the ICDW weights can be
        # calculated from a single state, so we will take both fields at their
        # lowest value: 1000010000.
        # It should have energy
        # -2 * end_weight (from the first spins of each field)
        # + 2 * alignment_weight (from the domain wall of each field)
        # - 2 * 3 * alignment_weight (from the 3 pairs of aligned neighboring
        # |0>s in each field)
        # - 2 * end_weight (from the last spin of each field)
        # = -2 * (2 * end_weight + 2 * alignment_weight)
        # = -4 * (end_weight + alignment_weight)
        base_energy =  -4.0 * (end_weight + alignment_weight)
        field_step_squared = self.field_step_in_GeV * self.field_step_in_GeV
        radius_step_squared = (
            self.radius_step_in_inverse_GeV * self.radius_step_in_inverse_GeV
        )
        extra_for_difference_of_one = (
            (0.5 * field_step_squared) / radius_step_squared
        )
        extra_for_difference_of_two = extra_for_difference_of_one * 4.0
        extra_for_difference_of_three = extra_for_difference_of_one * 9.0
        expected_bitstrings_to_energies = {
            "1000010000": base_energy,
            "1000011000": base_energy + extra_for_difference_of_one,
            "1000011100": base_energy + extra_for_difference_of_two,
            "1000011110": base_energy + extra_for_difference_of_three,
            "1100010000": base_energy + extra_for_difference_of_one,
            "1100011000": base_energy,
            "1100011100": base_energy + extra_for_difference_of_one,
            "1100011110": base_energy + extra_for_difference_of_two,
            "1110010000": base_energy + extra_for_difference_of_two,
            "1110011000": base_energy + extra_for_difference_of_one,
            "1110011100": base_energy,
            "1110011110": base_energy + extra_for_difference_of_one,
            "1111010000": base_energy + extra_for_difference_of_three,
            "1111011000": base_energy + extra_for_difference_of_two,
            "1111011100": base_energy + extra_for_difference_of_one,
            "1111011110": base_energy
        }
        self.assertEqual(
            len(expected_bitstrings_to_energies),
            len(actual_bitstrings_to_energies),
            "expected 4 * 4 results with valid bitstrings"
        )
        self.assertAlmostEqual(
            expected_bitstrings_to_energies["1000010000"],
            actual_bitstrings_to_energies["1000010000"],
            msg="expected only base energy for (0, 0)"
        )
        self.assertAlmostEqual(
            expected_bitstrings_to_energies["1000011000"],
            actual_bitstrings_to_energies["1000011000"],
            msg="expected base energy plus 0.5 * 1^2 * step^2 for (0, 1)"
        )
        self.assertAlmostEqual(
            expected_bitstrings_to_energies["1000011100"],
            actual_bitstrings_to_energies["1000011100"],
            msg="expected base energy plus 0.5 * 2^2 * step^2 for (0, 2)"
        )
        self.assertAlmostEqual(
            expected_bitstrings_to_energies["1000011110"],
            actual_bitstrings_to_energies["1000011110"],
            msg="expected base energy plus 0.5 * 3^2 * step^2 for (0, 3)"
        )
        self.assertEqual(
            expected_bitstrings_to_energies,
            actual_bitstrings_to_energies,
            "expected correct differences for all valid bitstrings"
        )

    def _set_up_fields(
            self,
            number_of_values_for_field
        ) -> Tuple[FieldAtPoint, FieldAtPoint]:
        return (
            FieldAtPoint(
                field_name=self.field_name,
                spatial_point_identifier="l",
                number_of_values_for_field=number_of_values_for_field,
                field_step_in_GeV=self.field_step_in_GeV
            ),
            FieldAtPoint(
                field_name=self.field_name,
                spatial_point_identifier="u",
                number_of_values_for_field=number_of_values_for_field,
                field_step_in_GeV=self.field_step_in_GeV
            )
        )


if __name__ == "__main__":
    unittest.main()
