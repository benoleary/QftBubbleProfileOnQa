import unittest
from hamiltonian.field import Field

class TestField(unittest.TestCase):
    def test_all_valid_strengths_for_only_domain_wall_conditions(self):
        test_field = \
            Field(
                "t",
                "x0",
                3
            )
        self.assertEqual(
            ["t_x0_0", "t_x0_1", "t_x0_2"],
            test_field.binary_variable_names,
            "incorrect names for binary variables"
        )

if __name__ == "__main__":
    unittest.main()
