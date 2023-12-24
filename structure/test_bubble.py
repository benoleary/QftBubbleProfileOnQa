import unittest
from configuration.configuration import DiscreteConfiguration
from structure.bubble import BubbleProfile

class TestBubbleProfile(unittest.TestCase):
    def test_spatial_identifiers_have_same_length(self):
        test_configuration = DiscreteConfiguration(
            first_field_name="f",
            number_of_spatial_steps=100,
            spatial_step_in_inverse_GeV=1.0,
            field_step_in_GeV=1.0,
            potential_in_quartic_GeV_per_field_step=[0.0, 1.0, 2.0]
        )
        test_bubble_profile = BubbleProfile(test_configuration)
        actual_spatial_identifiers = [
            p.spatial_point_identifier
            for p in test_bubble_profile.fields_at_points
        ]
        # There are 101 values if there are 100 steps.
        expected_spatial_identifiers = [
            "r{0:03}".format(i) for i in range(101)
        ]
        self.assertEqual(
            actual_spatial_identifiers,
            expected_spatial_identifiers,
            "incorrect text for spatial identifiers"
        )
        actual_lengths = [len(s) for s in actual_spatial_identifiers]
        self.assertEqual(
            actual_lengths,
            [4 for _ in range(101)],
            "incorrect length(s) for spatial identifiers"
        )

    def test_weights_for_monotonic_potential(self):
        field_step_in_GeV = 1.0
        test_configuration = DiscreteConfiguration(
            first_field_name="f",
            number_of_spatial_steps=1,
            spatial_step_in_inverse_GeV=1.0,
            field_step_in_GeV=field_step_in_GeV,
            potential_in_quartic_GeV_per_field_step=[-1.5, 0.9, 5.3]
        )
        test_bubble_profile = BubbleProfile(test_configuration)
        actual_spin_biases = test_bubble_profile.spin_biases

        # There are 2 points, each with 3 values for the field (so bitstrings
        # 1000, 1100, and 1110). The maximum potential difference is 6.8, so the
        # alignment weight should be 13.6 and the end weight should be 27.2.
        # Each "variable" spin should have a weight equal to half the difference
        # between its potential and the potential of the previous spin.
        expected_end_weight = 27.2
        expected_alignment_weight = -13.6

        # The weight is multiplied by the square of the field step and the
        # inverse square of the radius step but both are 1.
        expected_kinetic_weight = 0.125

        lower_index_expected_potential_difference = -1.2
        upper_index_expected_potential_difference = -2.2
        expected_linear_biases = {
            "f_r0_0": expected_end_weight,
            "f_r0_1": lower_index_expected_potential_difference,
            "f_r0_2": upper_index_expected_potential_difference,
            "f_r0_3": -expected_end_weight,
            "f_r1_0": expected_end_weight,
            "f_r1_1": lower_index_expected_potential_difference,
            "f_r1_2": upper_index_expected_potential_difference,
            "f_r1_3": -expected_end_weight
        }
        expected_quadratic_biases = {
            # There are four pairs which should have only the ICDW alignment
            # weight.
            ("f_r0_0", "f_r0_1"): (
                expected_alignment_weight
            ),
            ("f_r0_2", "f_r0_3"): (
                expected_alignment_weight
            ),
            ("f_r1_0", "f_r1_1"): (
                expected_alignment_weight
            ),
            ("f_r1_2", "f_r1_3"): (
                expected_alignment_weight
            ),
            # There are two pairs which have the alignment weight as well as the
            # kinetic term weight. The rest of the pairs are just from the
            # kinetic term.
            ("f_r0_1", "f_r0_1"): expected_kinetic_weight,
            ("f_r0_1", "f_r0_2"): (
                expected_alignment_weight + expected_kinetic_weight
            ),
            ("f_r0_2", "f_r0_1"): expected_kinetic_weight,
            ("f_r0_2", "f_r0_2"): expected_kinetic_weight,
            ("f_r0_1", "f_r1_1"): -expected_kinetic_weight,
            ("f_r0_1", "f_r1_2"): -expected_kinetic_weight,
            ("f_r0_2", "f_r1_1"): -expected_kinetic_weight,
            ("f_r0_2", "f_r1_2"): -expected_kinetic_weight,
            ("f_r1_1", "f_r0_1"): -expected_kinetic_weight,
            ("f_r1_1", "f_r0_2"): -expected_kinetic_weight,
            ("f_r1_2", "f_r0_1"): -expected_kinetic_weight,
            ("f_r1_2", "f_r0_2"): -expected_kinetic_weight,
            ("f_r1_1", "f_r1_1"): expected_kinetic_weight,
            ("f_r1_1", "f_r1_2"): (
                expected_alignment_weight + expected_kinetic_weight
            ),
            ("f_r1_2", "f_r1_1"): expected_kinetic_weight,
            ("f_r1_2", "f_r1_2"): expected_kinetic_weight
        }

        self.assertEqual(
            actual_spin_biases.linear_biases.keys(),
            expected_linear_biases.keys(),
            "incorrect linear bias variable names"
        )
        for variable_name in actual_spin_biases.linear_biases.keys():
            self.assertAlmostEqual(
                expected_linear_biases[variable_name],
                actual_spin_biases.linear_biases.get(variable_name, 0.0),
                msg=f"incorrect linear weight for {variable_name}"
            )

        self.assertEqual(
            actual_spin_biases.quadratic_biases.keys(),
            expected_quadratic_biases.keys(),
            "incorrect quadratic bias variable names"
        )
        for variable_name in actual_spin_biases.quadratic_biases.keys():
            self.assertAlmostEqual(
                expected_quadratic_biases[variable_name],
                actual_spin_biases.quadratic_biases.get(variable_name, 0.0),
                msg=f"incorrect quadratic weight for {variable_name}"
            )


if __name__ == "__main__":
    unittest.main()
