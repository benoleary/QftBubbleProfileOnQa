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

if __name__ == "__main__":
    unittest.main()
