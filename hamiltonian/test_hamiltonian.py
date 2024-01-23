from typing import Tuple
import pytest
from configuration.configuration import QftModelConfiguration
from hamiltonian.field import FieldAtPoint, FieldDefinition
from hamiltonian.hamiltonian import AnnealerHamiltonian
from hamiltonian.spin import SpinHamiltonian
from minimization.sampling import SamplerHandler, SamplePovider
from minimization.spin import SpinSamplerHandler
import minimization.variable
from structure.domain_wall import DomainWallWeighter
from structure.point import ProfileAtPoint
from structure.spin import SpinDomainWallWeighter



_field_step_in_GeV = 7.0
_true_vacuum_value_in_GeV=-10.75
_radius_step_in_inverse_GeV = 0.25
# We want the field to be
# |1000> => _true_vacuum_value_in_GeV,
# |1100> => _true_vacuum_value_in_GeV + _field_step_in_GeV, and
# |1110> => _true_vacuum_value_in_GeV + (2 * _field_step_in_GeV),
# so the upper bound and also false vacuum is at
# (_true_vacuum_value_in_GeV
# + ((number_of_values_for_field - 1) * _field_step_in_GeV)).
_upper_bound = _true_vacuum_value_in_GeV + (2 * _field_step_in_GeV)
_field_definition = FieldDefinition(
    field_name="T",
    number_of_values=3,
    lower_bound_in_GeV=_true_vacuum_value_in_GeV,
    upper_bound_in_GeV=_upper_bound,
    true_vacuum_value_in_GeV=_true_vacuum_value_in_GeV,
    false_vacuum_value_in_GeV=_upper_bound
)
_model_configuration = QftModelConfiguration(
    first_field=_field_definition,
    potential_in_quartic_GeV_per_field_step=[[1.0, 2.0]]
)
_lower_radius_field = FieldAtPoint(
    field_definition=_field_definition,
    spatial_point_identifier="l"
)
_upper_radius_field = FieldAtPoint(
    field_definition=_field_definition,
    spatial_point_identifier="u"
)

_spin_Hamiltonian = SpinHamiltonian(
    _model_configuration
)
_spin_domain_wall_weighter = SpinDomainWallWeighter()
_spin_sampler_handler = SpinSamplerHandler()

class TestAnnealerHamiltonians():
    @pytest.mark.parametrize(
            "annealer_Hamiltonian", "domain_wall_weighter", "sampler_handler"
            [
                (
                    _spin_Hamiltonian,
                    _spin_sampler_handler,
                    _spin_domain_wall_weighter
                )
            ]
    )
    def test_all_valid_bitstrings_present_with_correct_energies(
        self,
        annealer_Hamiltonian: AnnealerHamiltonian,
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        """
        This tests that the exact solver finds the correct energies for each
        valid pair of ICDW configurations of a field at neighboring radius
        values.
        """
        maximum_variable_weight = (
            annealer_Hamiltonian.get_maximum_potential_difference()
            + annealer_Hamiltonian.get_maximum_kinetic_contribution(
                _radius_step_in_inverse_GeV
            )
        )
        domain_wall_alignment_weight = 2.0 * maximum_variable_weight
        domain_end_fixing_weight = 2.0 * domain_wall_alignment_weight

        profiles_at_points = [
            ProfileAtPoint(
                spatial_point_identifier="r",
                spatial_radius_in_inverse_GeV=_radius_step_in_inverse_GeV,
                first_field=_field_definition
            ) for _ in range(2)
        ]

        calculated_weights = domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=profiles_at_points,
            end_weight=domain_end_fixing_weight,
            alignment_weight=domain_wall_alignment_weight
        )
        calculated_weights.add(
            annealer_Hamiltonian.kinetic_weights(
                radius_step_in_inverse_GeV=_radius_step_in_inverse_GeV,
                nearer_center=_lower_radius_field,
                nearer_edger=_upper_radius_field,
                scaling_factor=1.0
            )
        )

        sample_provider = SamplePovider(
            sampler_name="exact",
            sampler_handler=sampler_handler
        )

        sampling_result = sample_provider.get_sample(calculated_weights)

        # We grab all the results under the energy which should only happen if
        # the spins violate the conditions of the Ising-chain domain wall.
        samples_under_penalty_weight = sampling_result.lowest(
            atol=(0.75 * domain_wall_alignment_weight)
        )
        actual_bitstrings_to_energies = (
            minimization.variable.bitstrings_to_energies(
                binary_variable_names=(
                    _lower_radius_field.binary_variable_names
                    + _upper_radius_field.binary_variable_names
                ),
                sample_set=samples_under_penalty_weight
            )
        )

        # The lowest energy is going to be dependent on whether we use spin or
        # bit variables, so we just calculate differences from the easiest
        # state: 1000010000, which should have the lowest energy anyway (jointly
        # with other states of zero difference between the fields).
        zero_zero_bitstring = "1000010000"
        assert (
            zero_zero_bitstring in actual_bitstrings_to_energies.keys()
        ), f"expected {zero_zero_bitstring} to be a valid state"
        base_energy = actual_bitstrings_to_energies[zero_zero_bitstring]
        field_step_squared = _field_step_in_GeV * _field_step_in_GeV
        radius_step_squared = (
            _radius_step_in_inverse_GeV * _radius_step_in_inverse_GeV
        )
        extra_for_difference_of_one = (
            (0.5 * field_step_squared) / radius_step_squared
        )
        extra_for_difference_of_two = extra_for_difference_of_one * 4.0
        extra_for_difference_of_three = extra_for_difference_of_one * 9.0
        expected_bitstrings_to_energies = {
            zero_zero_bitstring: base_energy,
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
        assert (
            len(expected_bitstrings_to_energies)
            == len(actual_bitstrings_to_energies)
        ), "expected 4 * 4 results with valid bitstrings"
        assert (
            pytest.approx(expected_bitstrings_to_energies[zero_zero_bitstring])
            == actual_bitstrings_to_energies[zero_zero_bitstring]
        ), "expected only base energy for (0, 0)"
        assert (
            pytest.approx(expected_bitstrings_to_energies["1000011000"])
            == actual_bitstrings_to_energies["1000011000"]
        ), "expected base energy plus 0.5 * 1^2 * step^2 for (0, 1)"
        assert (
            pytest.approx(expected_bitstrings_to_energies["1000011100"])
            == actual_bitstrings_to_energies["1000011100"]
        ), "expected base energy plus 0.5 * 2^2 * step^2 for (0, 2)"
        assert (
            pytest.approx(expected_bitstrings_to_energies["1000011110"])
            == actual_bitstrings_to_energies["1000011110"]
        ), "expected base energy plus 0.5 * 3^2 * step^2 for (0, 3)"
        assert (
            expected_bitstrings_to_energies == actual_bitstrings_to_energies
        ), "expected correct differences for all valid bitstrings"

    def _set_up_fields(
            self,
            number_of_values_for_field
        ) -> Tuple[FieldDefinition, FieldAtPoint, FieldAtPoint]:
        field_name = "T"
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
        single_field_definition = FieldDefinition(
            field_name=field_name,
            number_of_values=number_of_values_for_field,
            lower_bound_in_GeV=_true_vacuum_value_in_GeV,
            upper_bound_in_GeV=upper_bound,
            true_vacuum_value_in_GeV=_true_vacuum_value_in_GeV,
            false_vacuum_value_in_GeV=upper_bound
        )
        return (
            single_field_definition,
            FieldAtPoint(
                field_definition=single_field_definition,
                spatial_point_identifier="l"
            ),
            FieldAtPoint(
                field_definition=single_field_definition,
                spatial_point_identifier="u"
            )
        )
