from typing import Tuple
import pytest

from basis.field import FieldCollectionAtPoint, FieldDefinition
import basis.variable
from dynamics.hamiltonian import AnnealerHamiltonian
from dynamics.spin import SpinHamiltonian
from input.configuration import QftModelConfiguration
from minimization.sampling import SamplerHandler, SamplePovider
from minimization.spin import SpinSamplerHandler
from structure.domain_wall import DomainWallWeighter
from structure.spin import SpinDomainWallWeighter


# We keep the bounds at the vacua, so _lower_bound is also the value in GeV of
# the potential at the true vacuum.
_lower_bound = -10.75
_field_step_in_GeV = 7.0
_radius_step_in_inverse_GeV = 0.25
_number_of_field_values = 5
# We want the field to be
# |100000> => _lower_bound,
# |110000> => _lower_bound + _field_step_in_GeV, and
# |111000> => _lower_bound + (2 * _field_step_in_GeV),
# ...
# so the upper bound is
# (_lower_bound + ((_number_of_field_values - 1) * _field_step_in_GeV)).
_upper_bound = (
    _lower_bound + (_number_of_field_values * _field_step_in_GeV)
)
_field_definition = FieldDefinition(
    field_name="T",
    number_of_values=3,
    lower_bound_in_GeV=_lower_bound,
    upper_bound_in_GeV=_upper_bound,
    true_vacuum_value_in_GeV=_lower_bound,
    false_vacuum_value_in_GeV=_upper_bound
)


# The linear term should not matter.
def _linear_potential(field_value: int):
    return (2.0 * field_value) + 20.0


_discretized_linear_potential = [
    [_linear_potential(f) for f in range(_number_of_field_values)]
]

_linear_potential_configuration = QftModelConfiguration(
    first_field=_field_definition,
    potential_in_quartic_GeV_per_field_step=_discretized_linear_potential
)

# TODO: bit version
_spin_linear_potential = SpinHamiltonian(
    _linear_potential_configuration
)


# As ever, the linear term is irrelevant.
def _quadratic_potential(field_value: int):
    return (2.5 * (field_value - 5) * (field_value - 5)) + 10.0


_discretized_quadratic_potential = [
    [_quadratic_potential(f) for f in range(_number_of_field_values)]
]

_quadratic_potential_configuration = QftModelConfiguration(
    first_field=_field_definition,
    potential_in_quartic_GeV_per_field_step=_discretized_quadratic_potential
)

# TODO: bit version
_spin_quadratic_potential = SpinHamiltonian(
    _quadratic_potential_configuration
)

# TODO: bit versions
_spin_domain_wall_weighter = SpinDomainWallWeighter()
_spin_sampler_handler = SpinSamplerHandler()


# TODO: tests with second field
class TestAnnealerHamiltonians():
    @pytest.mark.parametrize(
            "annealer_Hamiltonian", "domain_wall_weighter", "sampler_handler"
            [
                (
                    # The potential does not matter for the test of the kinetic
                    # term, so we just use the simpler case.
                    _spin_linear_potential,
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
        domain_end_fixing_weight, domain_wall_alignment_weight = (
            self._get_end_and_alignment_weights(annealer_Hamiltonian)
        )

        profiles_at_points = [
            FieldCollectionAtPoint(
                spatial_point_identifier=f"r{i}",
                spatial_radius_in_inverse_GeV=_radius_step_in_inverse_GeV,
                first_field=_field_definition
            ) for i in range(2)
        ]

        lower_radius_field = profiles_at_points[0].first_field
        upper_radius_field = profiles_at_points[1].first_field

        calculated_weights = domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=profiles_at_points,
            end_weight=domain_end_fixing_weight,
            alignment_weight=domain_wall_alignment_weight
        )
        calculated_weights.add(
            annealer_Hamiltonian.kinetic_weights(
                radius_step_in_inverse_GeV=_radius_step_in_inverse_GeV,
                nearer_center=lower_radius_field,
                nearer_edger=upper_radius_field,
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
            basis.variable.bitstrings_to_energies(
                binary_variable_names=(
                    lower_radius_field.binary_variable_names
                    + upper_radius_field.binary_variable_names
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

    @pytest.mark.parametrize(
            "annealer_Hamiltonian", "domain_wall_weighter", "sampler_handler"
            [
                (
                    _spin_linear_potential,
                    _spin_sampler_handler,
                    _spin_domain_wall_weighter
                )
            ]
    )
    def test_linear_potential_minimized_correctly(
        self,
        annealer_Hamiltonian: AnnealerHamiltonian,
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        bitstrings_to_energies = (
            self._get_bitstrings_to_energies_for_potential_at_single_point(
                annealer_Hamiltonian=annealer_Hamiltonian,
                domain_wall_weighter=domain_wall_weighter,
                sampler_handler=sampler_handler
            )
        )

        assert 1 == len(bitstrings_to_energies), "expected only one minimum"
        actual_solution_bitstring = next(iter(bitstrings_to_energies.keys()))
        assert (
            "10000000" == actual_solution_bitstring
        ), "expected domain wall completely to the left"

    @pytest.mark.parametrize(
            "annealer_Hamiltonian", "domain_wall_weighter", "sampler_handler"
            [
                (
                    _spin_quadratic_potential,
                    _spin_sampler_handler,
                    _spin_domain_wall_weighter
                )
            ]
    )
    def test_quadratic_potential_minimized_correctly(
        self,
        annealer_Hamiltonian: AnnealerHamiltonian,
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        bitstrings_to_energies = (
            self._get_bitstrings_to_energies_for_potential_at_single_point(
                annealer_Hamiltonian=annealer_Hamiltonian,
                domain_wall_weighter=domain_wall_weighter,
                sampler_handler=sampler_handler
            )
        )

        assert 1 == len(bitstrings_to_energies), "expected only one minimum"
        actual_solution_bitstring = next(iter(bitstrings_to_energies.keys()))
        assert (
            "11111100" == actual_solution_bitstring
        ), "expected 5 1s between fixed 1st bit and domain wall"

    def _get_end_and_alignment_weights(
            self,
            annealer_Hamiltonian: AnnealerHamiltonian
    ) -> Tuple[float, float]:
        maximum_variable_weight = (
            annealer_Hamiltonian.get_maximum_potential_difference()
            + annealer_Hamiltonian.get_maximum_kinetic_contribution(
                _radius_step_in_inverse_GeV
            )
        )
        domain_wall_alignment_weight = 2.0 * maximum_variable_weight
        domain_end_fixing_weight = 2.0 * domain_wall_alignment_weight
        return (domain_end_fixing_weight, domain_wall_alignment_weight)

    def _get_bitstrings_to_energies_for_potential_at_single_point(
            self,
            annealer_Hamiltonian: AnnealerHamiltonian,
            domain_wall_weighter: DomainWallWeighter,
            sampler_handler: SamplerHandler
    ):
        domain_end_fixing_weight, domain_wall_alignment_weight = (
            self._get_end_and_alignment_weights(annealer_Hamiltonian)
        )

        # One point is sufficient.
        profiles_at_points = [
            FieldCollectionAtPoint(
                spatial_point_identifier="r",
                spatial_radius_in_inverse_GeV=_radius_step_in_inverse_GeV,
                first_field=_field_definition
            )
        ]

        single_field = profiles_at_points[0].first_field

        calculated_weights = domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=profiles_at_points,
            end_weight=domain_end_fixing_weight,
            alignment_weight=domain_wall_alignment_weight
        )
        calculated_weights.add(
            annealer_Hamiltonian.potential_weights(
                first_field=single_field,
                scaling_factor=1.0
            )
        )

        sample_provider = SamplePovider(
            sampler_name="exact",
            sampler_handler=sampler_handler
        )

        sampling_result = sample_provider.get_sample(calculated_weights)

        lowest_energy = sampling_result.lowest(rtol=0.01, atol=0.1)

        return basis.variable.bitstrings_to_energies(
            binary_variable_names=single_field.binary_variable_names,
            sample_set=lowest_energy
        )
