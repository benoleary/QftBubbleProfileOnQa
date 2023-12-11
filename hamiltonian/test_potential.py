from typing import Callable, Tuple
import unittest
from dimod import ExactSolver
import minimization.variable
from minimization.weight import BiasAccumulator
from configuration.configuration import DiscreteConfiguration
from structure.bubble import BubbleProfile
from hamiltonian.field import FieldAtPoint
import hamiltonian.potential

class TestSingleFieldPotential(unittest.TestCase):
    # In this case, the linear term is irrelevant.
    def test_proportional_to_field_value(self):
        def linear_potential(field_value: int):
            return (2.5 * field_value) + 10.0
        _, actual_weights = self._for_single_test_field(
            single_field_potential=linear_potential,
            number_of_potential_values=5
        )
        actual_linear_weights = actual_weights.linear_biases

        # The weights are actually just the differences which each |0> flipping
        # to a |1> brings (with a factor of -0.5 because the spin flip itself
        # brings a factor of (-1)-(+1) = -2).
        expected_linear_weights = {
            "T_r0_1": -1.25,
            "T_r0_2": -1.25,
            "T_r0_3": -1.25,
            "T_r0_4": -1.25,
        }
        self.assertEqual(
            expected_linear_weights.keys(),
            actual_linear_weights.keys()
        )
        for expected_key in expected_linear_weights.keys():
            self.assertAlmostEqual(
                expected_linear_weights[expected_key],
                actual_linear_weights.get(expected_key, 0.0),
                msg=f"incorrect linear weight for {expected_key}"
            )

    def test_linear_potential_minimized_correctly(self):
        # In this case as well, the linear term is irrelevant.
        def linear_potential(field_value: int):
            return (2.0 * field_value) + 20.0
        test_field, potential_weights = self._for_single_test_field(
            single_field_potential=linear_potential,
            number_of_potential_values=7
        )
        test_sampler = ExactSolver()
        end_weight = 10.0
        alignment_weight = 3.5
        weight_accumulator = test_field.domain_wall_weights(
                end_spin_weight=end_weight,
                spin_alignment_weight=alignment_weight
            )
        weight_accumulator.add(potential_weights)

        sampling_result = test_sampler.sample_ising(
            h=weight_accumulator.linear_biases,
            J=weight_accumulator.quadratic_biases
        )
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        bitstrings_to_energies = {
            minimization.variable.as_bitstring(
                spin_variable_names=test_field.binary_variable_names,
                spin_mapping=s
            ): e
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

    def test_quadratic_potential_minimized_correctly(self):
        # As ever, the linear term is irrelevant.
        def quadratic_potential(field_value: int):
            return (2.5 * (field_value - 5) * (field_value - 5)) + 10.0
        test_field, potential_weights = self._for_single_test_field(
            single_field_potential=quadratic_potential,
            number_of_potential_values=7
        )
        test_sampler = ExactSolver()
        end_weight = 10.0
        alignment_weight = 3.5
        weight_accumulator = test_field.domain_wall_weights(
                end_spin_weight=end_weight,
                spin_alignment_weight=alignment_weight
            )
        weight_accumulator.add(potential_weights)

        sampling_result = test_sampler.sample_ising(
            h=weight_accumulator.linear_biases,
            J=weight_accumulator.quadratic_biases
        )
        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)
        bitstrings_to_energies = {
            minimization.variable.as_bitstring(
                spin_variable_names=test_field.binary_variable_names,
                spin_mapping=s
            ): e
            for s, e in [(d.sample, d.energy) for d in lowest_energy.data()]
        }

        self.assertEqual(
            1,
            len(bitstrings_to_energies),
            "expected only one minimum"
        )
        actual_solution_bitstring = next(iter(bitstrings_to_energies.keys()))
        self.assertEqual(
            "11111100",
            actual_solution_bitstring,
            "expected 5 1s between fixed 1st bit and domain wall"
        )

    def _for_single_test_field(
            self,
            *,
            single_field_potential: Callable[[int], float],
            number_of_potential_values: int
        ) -> Tuple[FieldAtPoint, BiasAccumulator]:
        discretized_potential = [
            single_field_potential(f) for f in range(number_of_potential_values)
        ]
        test_configuration = DiscreteConfiguration(
            first_field_name="T",
            number_of_spatial_steps=1,
            spatial_step_in_inverse_GeV=1.0,
            field_step_in_GeV=1.0,
            potential_in_quartic_GeV_per_field_step=discretized_potential
        )
        test_bubble_profile = BubbleProfile(test_configuration)
        test_field = test_bubble_profile.fields_at_points[0].first_field

        potential_weights = hamiltonian.potential.weights_for(
            test_configuration,
            test_field
        )
        return (test_field, potential_weights)
