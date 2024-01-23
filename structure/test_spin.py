import pytest

from basis.field import FieldAtPoint, FieldDefinition
from structure.spin import SpinDomainWallWeighter


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

        actual_weights = SpinDomainWallWeighter().weights_for_domain_walls(
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
            expected_linear_weights == actual_weights.linear_weights
        ), "incorrect weights for linear biases"
        assert (
            expected_quadratic_weights == actual_weights.quadratic_weights
        ), "incorrect weights for quadratic biases"

    @pytest.mark.parametrize(
            "number_of_ones",
            [
                (1,),
                (2,),
                (3,),
                (4,),
                (5,),
                # We also test the negative input convention.
                (-1,),
                (-2,),
                (-3,),
                (-4,),
                (-5,)
            ]
    )
    def test_fixing_value(self, number_of_ones):
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

        expected_linear_weights = {
            "t_x_0": fixing_weight,
            "t_x_1": (
                fixing_weight if (
                    (number_of_ones >= 2) or (0 > number_of_ones >= -4)
                )
                else -fixing_weight
            ),
            "t_x_2": (
                fixing_weight if (
                    (number_of_ones >= 3) or (0 > number_of_ones >= -3)
                )
                else -fixing_weight
            ),
            "t_x_3": (
                fixing_weight if (
                    (number_of_ones >= 4) or (0 > number_of_ones >= -2)
                )
                else -fixing_weight
            ),
            "t_x_4": (
                fixing_weight if (
                    (number_of_ones >= 5) or (number_of_ones == -1)
                )
                else -fixing_weight
            ),
            "t_x_5": -fixing_weight
        }

        # All the weights should be exactly representable in binary so
        # we can make floating-point number comparisons without needing
        # a tolerance.
        assert (
            expected_linear_weights == spin_biases.linear_weights
        ), "incorrect linear weights"
        assert (
            {} == spin_biases.quadratic_weights
        ), "incorrect quadratic weights"
