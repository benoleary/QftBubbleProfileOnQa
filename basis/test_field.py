from basis.field import FieldAtPoint, FieldDefinition


class TestFieldAtPoint():
    def test_binary_variable_names_constructed_correctly(self):
        test_field = FieldAtPoint(
            field_definition=FieldDefinition(
                field_name="t",
                number_of_values=2,
                lower_bound_in_GeV=0.0,
                upper_bound_in_GeV=1.0,
                true_vacuum_value_in_GeV=0.0,
                false_vacuum_value_in_GeV=1.0
            ),
            spatial_point_identifier="x0"
        )
        assert (
            ["t_x0_0", "t_x0_1", "t_x0_2"] == test_field.binary_variable_names
        ), "incorrect names for spin variables"

    def test_steps_correctly_calculated(self):
        test_field = FieldAtPoint(
            field_definition=FieldDefinition(
                field_name="t",
                number_of_values=5,
                lower_bound_in_GeV=-1.0,
                upper_bound_in_GeV=2.0,
                true_vacuum_value_in_GeV=-0.9,
                false_vacuum_value_in_GeV=1.24
            ),
            spatial_point_identifier="x"
        )
        # The field values are all exactly representable in binary and decimal
        # so we do not need any tolerance for these comparisons.
        assert (
            0.75 == test_field.field_definition.step_in_GeV
        ), "expected 5 values around 4 steps making a range of 3 GeV: 0.75/step"
        assert (
            0 == test_field.field_definition.true_vacuum_value_in_steps
        ), "true vacuum at 0.9 should have been closer to 0 steps than to 1"
        assert (
            3 == test_field.field_definition.false_vacuum_value_in_steps
        ), "false vacuum at 1.74 should have been closer to 3 steps than to 2"
