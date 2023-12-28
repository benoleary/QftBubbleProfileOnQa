import pytest
from configuration.configuration import DiscreteConfiguration
from structure.bubble import BubbleProfile


class TestBubbleProfile():
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
        assert (
            actual_spatial_identifiers == expected_spatial_identifiers
        ), "incorrect text for spatial identifiers"
        actual_lengths = [len(s) for s in actual_spatial_identifiers]
        assert (
            actual_lengths == [4 for _ in range(101)]
        ), "incorrect length(s) for spatial identifiers"

    def test_weights_for_monotonic_potential(self):
        field_step_in_GeV = 1.0
        test_configuration = DiscreteConfiguration(
            first_field_name="f",
            number_of_spatial_steps=4,
            spatial_step_in_inverse_GeV=1.0,
            field_step_in_GeV=field_step_in_GeV,
            potential_in_quartic_GeV_per_field_step=[-1.5, 0.9, 5.3]
        )
        test_bubble_profile = BubbleProfile(test_configuration)
        actual_spin_biases = test_bubble_profile.spin_biases

        # There are five points, each with three values for the field (so
        # bitstrings 1000, 1100, and 1110). The maximum potential difference is
        # 6.8 and the maximum kinetic term is 0.5 * 2^2, so the alignment weight
        # should be 17.6 and the end weight should be 35.2.
        # Each "variable" spin should have a weight equal to half the difference
        # between its potential and the potential of the previous spin.
        expected_spin_weight = 35.2
        expected_alignment_weight = -17.6

        # The weight is multiplied by the square of the field step and the
        # inverse square of the radius step but both are 1.
        expected_kinetic_weight = 0.125

        lower_index_expected_potential_difference = -1.2
        upper_index_expected_potential_difference = -2.2
        expected_linear_biases = {
            # The center field should have weights to keep it at 1000
            "f_r0_0": expected_spin_weight,
            "f_r0_1": -expected_spin_weight,
            "f_r0_2": -expected_spin_weight,
            "f_r0_3": -expected_spin_weight,
            # The field 1 step from the center should have variation in its
            # middle spins dependent on the potential.
            "f_r1_0": expected_spin_weight,
            "f_r1_1": lower_index_expected_potential_difference,
            "f_r1_2": upper_index_expected_potential_difference,
            "f_r1_3": -expected_spin_weight,
            # The field 2 steps from the center should have variation in its
            # middle spins dependent on the potential.
            "f_r2_0": expected_spin_weight,
            "f_r2_1": lower_index_expected_potential_difference,
            "f_r2_2": upper_index_expected_potential_difference,
            "f_r2_3": -expected_spin_weight,
            # The field 3 steps from the center should have variation in its
            # middle spins dependent on the potential.
            "f_r3_0": expected_spin_weight,
            "f_r3_1": lower_index_expected_potential_difference,
            "f_r3_2": upper_index_expected_potential_difference,
            "f_r3_3": -expected_spin_weight,
            # The edge field should have weights to keep it at 1110
            "f_r4_0": expected_spin_weight,
            "f_r4_1": expected_spin_weight,
            "f_r4_2": expected_spin_weight,
            "f_r4_3": -expected_spin_weight
        }
        expected_quadratic_biases = {
            # There are six pairs which should have only the ICDW alignment
            # weight.
            ("f_r1_0", "f_r1_1"): expected_alignment_weight,
            ("f_r1_2", "f_r1_3"): expected_alignment_weight,
            ("f_r2_0", "f_r2_1"): expected_alignment_weight,
            ("f_r2_2", "f_r2_3"): expected_alignment_weight,
            ("f_r3_0", "f_r3_1"): expected_alignment_weight,
            ("f_r3_2", "f_r3_3"): expected_alignment_weight,
            # There are six pairs which have the alignment weight as well as the
            # kinetic term weight. The rest of the pairs are just from the
            # kinetic term.
            # The r0 field with itself is all positive.
            ("f_r0_1", "f_r0_1"): expected_kinetic_weight,
            ("f_r0_1", "f_r0_2"): expected_kinetic_weight,
            ("f_r0_2", "f_r0_1"): expected_kinetic_weight,
            ("f_r0_2", "f_r0_2"): expected_kinetic_weight,
            # The r0 field with r1 is all negative.
            ("f_r0_1", "f_r1_1"): -expected_kinetic_weight,
            ("f_r0_1", "f_r1_2"): -expected_kinetic_weight,
            ("f_r0_2", "f_r1_1"): -expected_kinetic_weight,
            ("f_r0_2", "f_r1_2"): -expected_kinetic_weight,
            # The r1 field with r0 is all negative.
            ("f_r1_1", "f_r0_1"): -expected_kinetic_weight,
            ("f_r1_1", "f_r0_2"): -expected_kinetic_weight,
            ("f_r1_2", "f_r0_1"): -expected_kinetic_weight,
            ("f_r1_2", "f_r0_2"): -expected_kinetic_weight,
            # The r1 field with itself is all positive, and gets contributions
            # from the difference from r0 and from r2, hence the factors of 2.
            # There is also one correlation from the domain wall weights.
            ("f_r1_1", "f_r1_1"): 2.0 * expected_kinetic_weight,
            ("f_r1_1", "f_r1_2"): (
                expected_alignment_weight + (2.0 * expected_kinetic_weight)
            ),
            ("f_r1_2", "f_r1_1"): 2.0 * expected_kinetic_weight,
            ("f_r1_2", "f_r1_2"): 2.0 * expected_kinetic_weight,
            # The r1 field with r2 is all negative.
            ("f_r1_1", "f_r2_1"): -expected_kinetic_weight,
            ("f_r1_1", "f_r2_2"): -expected_kinetic_weight,
            ("f_r1_2", "f_r2_1"): -expected_kinetic_weight,
            ("f_r1_2", "f_r2_2"): -expected_kinetic_weight,
            # The r2 field with r1 is all negative.
            ("f_r2_1", "f_r1_1"): -expected_kinetic_weight,
            ("f_r2_1", "f_r1_2"): -expected_kinetic_weight,
            ("f_r2_2", "f_r1_1"): -expected_kinetic_weight,
            ("f_r2_2", "f_r1_2"): -expected_kinetic_weight,
            # The r2 field with itself is similar to r1 with itself.
            ("f_r2_1", "f_r2_1"): 2.0 * expected_kinetic_weight,
            ("f_r2_1", "f_r2_2"): (
                expected_alignment_weight + (2.0 * expected_kinetic_weight)
            ),
            ("f_r2_2", "f_r2_1"): 2.0 * expected_kinetic_weight,
            ("f_r2_2", "f_r2_2"): 2.0 * expected_kinetic_weight,
            # The r2 field with r3 is all negative.
            ("f_r2_1", "f_r3_1"): -expected_kinetic_weight,
            ("f_r2_1", "f_r3_2"): -expected_kinetic_weight,
            ("f_r2_2", "f_r3_1"): -expected_kinetic_weight,
            ("f_r2_2", "f_r3_2"): -expected_kinetic_weight,
            # The r3 field with r2 is all negative.
            ("f_r3_1", "f_r2_1"): -expected_kinetic_weight,
            ("f_r3_1", "f_r2_2"): -expected_kinetic_weight,
            ("f_r3_2", "f_r2_1"): -expected_kinetic_weight,
            ("f_r3_2", "f_r2_2"): -expected_kinetic_weight,
            # The r3 field with itself is similar to r1 with itself.
            ("f_r3_1", "f_r3_1"): 2.0 * expected_kinetic_weight,
            ("f_r3_1", "f_r3_2"): (
                expected_alignment_weight + (2.0 * expected_kinetic_weight)
            ),
            ("f_r3_2", "f_r3_1"): 2.0 * expected_kinetic_weight,
            ("f_r3_2", "f_r3_2"): 2.0 * expected_kinetic_weight,
            # The r3 field with r4 is all negative.
            ("f_r3_1", "f_r4_1"): -expected_kinetic_weight,
            ("f_r3_1", "f_r4_2"): -expected_kinetic_weight,
            ("f_r3_2", "f_r4_1"): -expected_kinetic_weight,
            ("f_r3_2", "f_r4_2"): -expected_kinetic_weight,
            # The r4 field with r3 is all negative.
            ("f_r4_1", "f_r3_1"): -expected_kinetic_weight,
            ("f_r4_1", "f_r3_2"): -expected_kinetic_weight,
            ("f_r4_2", "f_r3_1"): -expected_kinetic_weight,
            ("f_r4_2", "f_r3_2"): -expected_kinetic_weight,
            # The r4 field with itself is like r0.
            ("f_r4_1", "f_r4_1"): expected_kinetic_weight,
            ("f_r4_1", "f_r4_2"): expected_kinetic_weight,
            ("f_r4_2", "f_r4_1"): expected_kinetic_weight,
            ("f_r4_2", "f_r4_2"): expected_kinetic_weight
        }

        assert (
            actual_spin_biases.linear_biases.keys()
            == expected_linear_biases.keys()
        ), "incorrect linear bias variable names"
        for variable_name in actual_spin_biases.linear_biases.keys():
            assert (
                pytest.approx(expected_linear_biases[variable_name])
                == actual_spin_biases.linear_biases.get(variable_name, 0.0)
            ), f"incorrect linear weight for {variable_name}"

        assert (
            actual_spin_biases.quadratic_biases.keys()
            == expected_quadratic_biases.keys()
        ), "incorrect quadratic bias variable names"
        for variable_name in actual_spin_biases.quadratic_biases.keys():
            assert(
                pytest.approx(expected_quadratic_biases[variable_name])
                == actual_spin_biases.quadratic_biases.get(variable_name, 0.0)
            ), f"incorrect quadratic weight for {variable_name}"
