import unittest
from dimod import ExactSolver
from hamiltonian.field import FieldAtPoint
from hamiltonian.potential import SingleFieldPotential

class TestSingleFieldPotential(unittest.TestCase):
    def test_proportional_to_field_value(self):
        test_field = \
            FieldAtPoint(
                field_name = "t",
                spatial_point_identifier = "x",
                number_of_active_binary_variables = 4
            )
        # In this case, the linear term is irrelevant.
        test_potential = SingleFieldPotential(lambda f : (2.0 * f) + 10.0)

        actual_weight_matrix = \
            test_potential.get_weights_for_potential(test_field).weight_matrix

        # The weights are actually just the differences which each 0 flipping to
        # a 1 brings.
        expected_weights = {
            ("t_x_1", "t_x_1"): 0.5,
            ("t_x_2", "t_x_2"): 0.5,
            ("t_x_3", "t_x_3"): 0.5,
            ("t_x_4", "t_x_4"): 0.5,
        }
        self.assertEqual(
            expected_weights.keys(),
            actual_weight_matrix.keys()
        )
        for expected_key in expected_weights.keys():
            self.assertAlmostEqual(
                expected_weights[expected_key],
                actual_weight_matrix.get(expected_key, 0.0),
                msg = f"incorrect weight for {expected_key}"
            )

    def test_linear_potential_minimized_correctly(self):
        test_sampler = ExactSolver()
        test_field = \
            FieldAtPoint(
                field_name = "t",
                spatial_point_identifier = "x0",
                number_of_active_binary_variables = 6
            )
        end_weight = 10.0
        alignment_weight = 3.5
        weight_accumulator = \
            test_field.weights_for_ICDW(
                end_spin_weight = end_weight,
                spin_alignment_weight = alignment_weight
            )
        # In this case as well, the linear term is irrelevant.
        test_potential = SingleFieldPotential(lambda f : (2.0 * f) - 20.0)
        weight_accumulator.add(
            test_potential.get_weights_for_potential(test_field).weight_matrix
        )

        sampling_result = \
            test_sampler.sample_qubo(weight_accumulator.weight_matrix)
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        bitstrings_to_energies = {
            "".join([f"{s[n]}" for n in test_field.binary_variable_names]): e
            for s, e in [(d.sample, d.energy) for d in lowest_energy.data()]
        }

        self.assertEqual(
            1,
            len(bitstrings_to_energies),
            "expected only one minimum"
        )
        actual_solution_bitstring = next(iter(bitstrings_to_energies.keys()))
        self.assertEqual(
            "10000000",
            actual_solution_bitstring,
            "expected domain wall completely to the left"
        )
