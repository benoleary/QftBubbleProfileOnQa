from __future__ import annotations

import pytest

from input.configuration import QftModelConfiguration
from basis.field import FieldAtPoint, FieldCollectionAtPoint, FieldDefinition
from dynamics.spin import SpinHamiltonian


_field_step_in_GeV = 7.0
_true_vacuum_value_in_GeV=-10.75
_radius_step_in_inverse_GeV = 0.25
# The weight should be (1/8) * 7^2 * 4^2 = (49 * 2) = 98
_absolute_expected_weight = 98.0


class TestSpinHamiltonian():
    def test_kinetic_weights_for_low_resolution_fields(self):
        """
        This tests that the correct weights are generated using an almost
        minimal number of spins for the fields.
        """
        # Each field should have three valid bitstrings:
        # |1000>, |1100>, and |1110>.
        field_definition, lower_radius_field, upper_radius_field = (
            self._set_up_fields_for_kinetic_tests(3)
        )

        test_Hamiltonian = SpinHamiltonian(
            QftModelConfiguration(
                first_field=field_definition,
                potential_in_quartic_GeV_per_field_step=[[1.0, 2.0]]
            )
        )
        actual_weights = test_Hamiltonian.kinetic_weights(
            radius_step_in_inverse_GeV=_radius_step_in_inverse_GeV,
            nearer_center=lower_radius_field,
            nearer_edge=upper_radius_field,
            scaling_factor=1.0
        )

        # Only the spins which can vary should get weights, so there should be
        # no appearaces of T_l_0, T_l_3, T_r_0, or T_r_3.
        expected_quadratic_weights = {
            # The lower field with itself should be all positive
            ("T_l_1", "T_l_1"): _absolute_expected_weight,
            ("T_l_1", "T_l_2"): _absolute_expected_weight,
            ("T_l_2", "T_l_1"): _absolute_expected_weight,
            ("T_l_2", "T_l_2"): _absolute_expected_weight,
            # The lower field with the upper field should be all negative
            ("T_l_1", "T_u_1"): -_absolute_expected_weight,
            ("T_l_1", "T_u_2"): -_absolute_expected_weight,
            ("T_l_2", "T_u_1"): -_absolute_expected_weight,
            ("T_l_2", "T_u_2"): -_absolute_expected_weight,
            # The upper field with itself should be all positive
            ("T_u_1", "T_u_1"): _absolute_expected_weight,
            ("T_u_1", "T_u_2"): _absolute_expected_weight,
            ("T_u_2", "T_u_1"): _absolute_expected_weight,
            ("T_u_2", "T_u_2"): _absolute_expected_weight,
            # The upper field with the lower field should be all negative
            ("T_u_1", "T_l_1"): -_absolute_expected_weight,
            ("T_u_1", "T_l_2"): -_absolute_expected_weight,
            ("T_u_2", "T_l_1"): -_absolute_expected_weight,
            ("T_u_2", "T_l_2"): -_absolute_expected_weight
        }

        # All the weights should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        assert (
            {} == actual_weights.linear_weights
        ), "incorrect linear weights"
        assert (
            len(expected_quadratic_weights.keys())
            == len(actual_weights.quadratic_weights.keys())
        ), "incorrect number of keys for quadratic weights"
        assert (
            expected_quadratic_weights.keys()
            == actual_weights.quadratic_weights.keys()
        ), "incorrect keys for quadratic weights"
        assert (
            expected_quadratic_weights == actual_weights.quadratic_weights
        ), "incorrect quadratic weights"

    def test_kinetic_weights_for_sightly_higher_resolution_fields(self):
        """
        This tests that the correct weights are generated using a small but not
        minimal number of spins for the fields.
        """
        # Each field should have four valid bitstrings:
        # |10000>, |11000>, |11100>, and |11110>.
        field_definition, lower_radius_field, upper_radius_field = (
            self._set_up_fields_for_kinetic_tests(4)
        )
        test_Hamiltonian = SpinHamiltonian(
            QftModelConfiguration(
                first_field=field_definition,
                potential_in_quartic_GeV_per_field_step=[[1.0, 2.0]]
            )
        )
        actual_weights = test_Hamiltonian.kinetic_weights(
            radius_step_in_inverse_GeV=_radius_step_in_inverse_GeV,
            nearer_center=lower_radius_field,
            nearer_edge=upper_radius_field,
            scaling_factor=1.0
        )

        # Only the spins which can vary should get weights, so there should be
        # no appearaces of T_l_0, T_l_4, T_r_0, or T_r_4.
        expected_quadratic_weights = {
            # The lower field with itself should be all positive
            ("T_l_1", "T_l_1"): _absolute_expected_weight,
            ("T_l_1", "T_l_2"): _absolute_expected_weight,
            ("T_l_1", "T_l_3"): _absolute_expected_weight,
            ("T_l_2", "T_l_1"): _absolute_expected_weight,
            ("T_l_2", "T_l_2"): _absolute_expected_weight,
            ("T_l_2", "T_l_3"): _absolute_expected_weight,
            ("T_l_3", "T_l_1"): _absolute_expected_weight,
            ("T_l_3", "T_l_2"): _absolute_expected_weight,
            ("T_l_3", "T_l_3"): _absolute_expected_weight,
            # The lower field with the upper field should be all negative
            ("T_l_1", "T_u_1"): -_absolute_expected_weight,
            ("T_l_1", "T_u_2"): -_absolute_expected_weight,
            ("T_l_1", "T_u_3"): -_absolute_expected_weight,
            ("T_l_2", "T_u_1"): -_absolute_expected_weight,
            ("T_l_2", "T_u_2"): -_absolute_expected_weight,
            ("T_l_2", "T_u_3"): -_absolute_expected_weight,
            ("T_l_3", "T_u_1"): -_absolute_expected_weight,
            ("T_l_3", "T_u_2"): -_absolute_expected_weight,
            ("T_l_3", "T_u_3"): -_absolute_expected_weight,
            # The upper field with itself should be all positive
            ("T_u_1", "T_u_1"): _absolute_expected_weight,
            ("T_u_1", "T_u_2"): _absolute_expected_weight,
            ("T_u_1", "T_u_3"): _absolute_expected_weight,
            ("T_u_2", "T_u_1"): _absolute_expected_weight,
            ("T_u_2", "T_u_2"): _absolute_expected_weight,
            ("T_u_2", "T_u_3"): _absolute_expected_weight,
            ("T_u_3", "T_u_1"): _absolute_expected_weight,
            ("T_u_3", "T_u_2"): _absolute_expected_weight,
            ("T_u_3", "T_u_3"): _absolute_expected_weight,
            # The upper field with the lower field should be all negative
            ("T_u_1", "T_l_1"): -_absolute_expected_weight,
            ("T_u_1", "T_l_2"): -_absolute_expected_weight,
            ("T_u_1", "T_l_3"): -_absolute_expected_weight,
            ("T_u_2", "T_l_1"): -_absolute_expected_weight,
            ("T_u_2", "T_l_2"): -_absolute_expected_weight,
            ("T_u_2", "T_l_3"): -_absolute_expected_weight,
            ("T_u_3", "T_l_1"): -_absolute_expected_weight,
            ("T_u_3", "T_l_2"): -_absolute_expected_weight,
            ("T_u_3", "T_l_3"): -_absolute_expected_weight
        }

        # All the weights should be exactly representable in binary so we can
        # make floating-point number comparisons without needing a tolerance.
        assert (
            {} == actual_weights.linear_weights
        ), "incorrect linear weights"
        assert (
            len(expected_quadratic_weights.keys())
            == len(actual_weights.quadratic_weights.keys())
        ), "incorrect number of keys for quadratic weights"
        assert (
            expected_quadratic_weights.keys()
            == actual_weights.quadratic_weights.keys()
        ), "incorrect keys for quadratic weights"
        assert (
            expected_quadratic_weights == actual_weights.quadratic_weights
        ), "incorrect quadratic weights"

    def test_weights_for_potential_proportional_to_field_value(self):
        # In this case, the linear term is irrelevant.
        def linear_potential(field_value: int):
            return (2.5 * field_value) + 10.0

        number_of_potential_values = 5
        field_definition = self._set_up_field_definition(
            number_of_potential_values
        )
        discretized_potential = [
            [linear_potential(f) for f in range(number_of_potential_values)]
        ]

        test_Hamiltonian = SpinHamiltonian(
            QftModelConfiguration(
                first_field=field_definition,
                potential_in_quartic_GeV_per_field_step=discretized_potential
            )
        )
        actual_weights = test_Hamiltonian.potential_weights(
            first_field=FieldAtPoint(
                field_definition=field_definition,
                spatial_point_identifier="r"
            ),
            scaling_factor=1.0
        )

        actual_linear_weights = actual_weights.linear_weights

        # The weights are actually just the differences which each |0> flipping
        # to a |1> brings (with a factor of -0.5 because the spin flip itself
        # brings a factor of (-1)-(+1) = -2).
        expected_linear_weights = {
            "T_r_1": -1.25,
            "T_r_2": -1.25,
            "T_r_3": -1.25,
            "T_r_4": -1.25,
        }
        assert expected_linear_weights.keys() == actual_linear_weights.keys()

        for expected_key in expected_linear_weights.keys():
            assert (
                pytest.approx(expected_linear_weights[expected_key])
                == actual_linear_weights.get(expected_key, 0.0)
            ), f"incorrect linear weight for {expected_key}"


    def _set_up_field_definition(
            self,
            number_of_values_for_field
    ) -> FieldDefinition:
        # For example, if number_of_values_for_field = 3, then we want the field
        # to be
        # |1000> => _true_vacuum_value_in_GeV,
        # |1100> => _true_vacuum_value_in_GeV + _field_step_in_GeV, and
        # |1110> => _true_vacuum_value_in_GeV + (2 * _field_step_in_GeV),
        # so the upper bound and also false vacuum is at
        # (_true_vacuum_value_in_GeV
        # + ((number_of_values_for_field - 1) * _field_step_in_GeV)).
        upper_bound = (
            _true_vacuum_value_in_GeV
            + ((number_of_values_for_field - 1) * _field_step_in_GeV)
        )
        return FieldDefinition(
            field_name="T",
            number_of_values=number_of_values_for_field,
            lower_bound_in_GeV=_true_vacuum_value_in_GeV,
            upper_bound_in_GeV=upper_bound,
            true_vacuum_value_in_GeV=_true_vacuum_value_in_GeV,
            false_vacuum_value_in_GeV=upper_bound
        )

    def _set_up_fields_for_kinetic_tests(
            self,
            number_of_values_for_field
    ) -> tuple[FieldDefinition, FieldCollectionAtPoint, FieldCollectionAtPoint]:
        single_field_definition = self._set_up_field_definition(
            number_of_values_for_field
        )
        return (
            single_field_definition,
            FieldCollectionAtPoint(
                spatial_point_identifier="l",
                spatial_radius_in_inverse_GeV=0.0,
                first_field=single_field_definition
            ),
            FieldCollectionAtPoint(
                spatial_point_identifier="u",
                spatial_radius_in_inverse_GeV=_radius_step_in_inverse_GeV,
                first_field=single_field_definition
            )
        )
