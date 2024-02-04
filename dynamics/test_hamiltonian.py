from __future__ import annotations
from typing import Callable, Optional

import pytest

from basis.field import FieldCollectionAtPoint, FieldDefinition
from dynamics.hamiltonian import AnnealerHamiltonian
from dynamics.bit import BitHamiltonian
from dynamics.spin import SpinHamiltonian
from input.configuration import QftModelConfiguration
from minimization.sampling import SamplerHandler, SampleProvider
from minimization.bit import BitSamplerHandler
from minimization.spin import SpinSamplerHandler
from structure.domain_wall import DomainWallWeighter
from structure.bit import BitDomainWallWeighter
from structure.spin import SpinDomainWallWeighter


def _as_differences_from_lowest_energy(
        *,
        bitstrings_to_absolute_energies: dict[str, float],
        absolute_tolerance_in_quartic_GeV: float
) -> dict[str, float]:
    lowest_energy = None
    for v in bitstrings_to_absolute_energies.values():
        if lowest_energy is None or v < lowest_energy:
            lowest_energy = v

    return {
        k: (v - lowest_energy)
        for k, v in bitstrings_to_absolute_energies.items()
        if (v - lowest_energy) < absolute_tolerance_in_quartic_GeV
    }


# We keep the bounds at the vacua, so _lower_bound is also the value in GeV of
# the potential at the true vacuum.
_lower_bound = -10.75
_field_step_in_GeV = 7.0
_radius_step_in_inverse_GeV = 0.25


# We want the field to be, for example for 5 values:
# |100000> => _lower_bound,
# |110000> => _lower_bound + _field_step_in_GeV, and
# |111000> => _lower_bound + (2 * _field_step_in_GeV),
# ...
# so the upper bound is
# (_lower_bound + ((_number_of_field_values - 1) * _field_step_in_GeV)).
def _get_test_field_definition(
        *,
        number_of_field_values: int,
        field_name: str = "T"
) -> FieldDefinition:
    upper_bound = (
        _lower_bound + ((number_of_field_values - 1) * _field_step_in_GeV)
    )
    return FieldDefinition(
        field_name=field_name,
        number_of_values=number_of_field_values,
        lower_bound_in_GeV=_lower_bound,
        upper_bound_in_GeV=upper_bound,
        true_vacuum_value_in_GeV=_lower_bound,
        false_vacuum_value_in_GeV=upper_bound
    )


def _zero_potential(field_value: int):
    return 0.0


# The linear term should not matter.
def _linear_single_field_potential(field_value: int):
    return (2.0 * field_value) + 20.0


# As ever, the linear term is irrelevant.
def _quadratic_single_field_potential(field_value: int):
    return (2.5 * (field_value - 5) * (field_value - 5)) + 10.0


def _get_single_field_potential_configuration(
        *,
        field_definition: FieldDefinition,
        potential_function: Callable[[int], float]
) -> QftModelConfiguration:
    return QftModelConfiguration(
        first_field=field_definition,
        potential_in_quartic_GeV_per_field_step=[
            [
                potential_function(f)
                for f in range(field_definition.number_of_values)
            ]
        ]
    )


def _get_spin_potential(
        model_configuration: QftModelConfiguration
) -> AnnealerHamiltonian:
    return SpinHamiltonian(model_configuration)


def _get_bit_potential(
        model_configuration: QftModelConfiguration
) -> AnnealerHamiltonian:
    return BitHamiltonian(model_configuration)


_spin_domain_wall_weighter = SpinDomainWallWeighter()
_spin_sampler_handler = SpinSamplerHandler()
_bit_domain_wall_weighter = BitDomainWallWeighter()
_bit_sampler_handler = BitSamplerHandler()


# TODO: tests with second field
class TestAnnealerHamiltonians():
    @pytest.mark.parametrize(
            "get_Hamiltonian, domain_wall_weighter, sampler_handler",
            [
                (
                    # The potential does not matter for the test of the kinetic
                    # term, so we just use the simpler case.
                    _get_spin_potential,
                    _spin_domain_wall_weighter,
                    _spin_sampler_handler
                ),
                (
                    _get_bit_potential,
                    _bit_domain_wall_weighter,
                    _bit_sampler_handler
                )
            ],
            ids=[
                "spin",
                "bit"
            ]
    )
    def test_all_valid_states_for_single_field_in_zero_potential(
        self,
        *,
        get_Hamiltonian: Callable[[QftModelConfiguration], AnnealerHamiltonian],
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        """
        This tests that the exact solver finds the correct energies for each
        valid pair of ICDW configurations of a field at neighboring radius
        values, removing any effect from the QFT potential by it being zero
        everywhere.
        """
        # We want the field to take values 0, 1, 2, and 3, leading to a maximum
        # difference of 3.
        number_of_field_values = 4
        field_definition = _get_test_field_definition(
            number_of_field_values=number_of_field_values
        )
        potential_configuration = _get_single_field_potential_configuration(
            field_definition=field_definition,
            potential_function=_zero_potential
        )
        annealer_Hamiltonian = get_Hamiltonian(potential_configuration)

        domain_end_fixing_weight, domain_wall_alignment_weight = (
            self._get_end_and_alignment_weights(annealer_Hamiltonian)
        )

        profiles_at_points = [
            FieldCollectionAtPoint(
                spatial_point_identifier=f"r{i}",
                spatial_radius_in_inverse_GeV=_radius_step_in_inverse_GeV,
                first_field=field_definition
            ) for i in range(2)
        ]

        at_lower_radius = profiles_at_points[0]
        at_upper_radius = profiles_at_points[1]

        calculated_weights = domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=profiles_at_points,
            end_weight=domain_end_fixing_weight,
            alignment_weight=domain_wall_alignment_weight
        )
        calculated_weights.add(
            annealer_Hamiltonian.kinetic_weights(
                radius_step_in_inverse_GeV=_radius_step_in_inverse_GeV,
                nearer_center=at_lower_radius,
                nearer_edge=at_upper_radius,
                scaling_factor=1.0
            )
        )

        sample_provider = SampleProvider(
            sampler_name="exact",
            sampler_handler=sampler_handler
        )
        sampling_result = sample_provider.get_sample(calculated_weights)

        actual_bitstrings_to_energies = (
            sample_provider.bitstrings_to_energies(
                binary_variable_names=(
                    at_lower_radius.first_field.binary_variable_names
                    + at_upper_radius.first_field.binary_variable_names
                ),
                sample_set=sampling_result
            )
        )

        # The lowest energy is going to be dependent on whether we use spin or
        # bit variables, so we just calculate differences from the easiest
        # state: 1000010000, which should have the lowest energy anyway (jointly
        # with other states of zero difference between the fields).
        # We grab all the results under the energy which should only happen if
        # the alignments violate the conditions of the Ising-chain domain wall.
        actual_bitstrings_to_differences = (
            _as_differences_from_lowest_energy(
                bitstrings_to_absolute_energies=actual_bitstrings_to_energies,
                absolute_tolerance_in_quartic_GeV=(
                    0.75 * domain_wall_alignment_weight
                )
            )
        )

        field_step_squared = _field_step_in_GeV * _field_step_in_GeV
        radius_step_squared = (
            _radius_step_in_inverse_GeV * _radius_step_in_inverse_GeV
        )
        extra_for_difference_of_one = (
            (0.5 * field_step_squared) / radius_step_squared
        )
        extra_for_difference_of_two = extra_for_difference_of_one * 4.0
        extra_for_difference_of_three = extra_for_difference_of_one * 9.0

        expected_bitstrings_to_differences = {
            "1000010000": 0.0,
            "1000011000": extra_for_difference_of_one,
            "1000011100": extra_for_difference_of_two,
            "1000011110": extra_for_difference_of_three,
            "1100010000": extra_for_difference_of_one,
            "1100011000": 0.0,
            "1100011100": extra_for_difference_of_one,
            "1100011110": extra_for_difference_of_two,
            "1110010000": extra_for_difference_of_two,
            "1110011000": extra_for_difference_of_one,
            "1110011100": 0.0,
            "1110011110": extra_for_difference_of_one,
            "1111010000": extra_for_difference_of_three,
            "1111011000": extra_for_difference_of_two,
            "1111011100": extra_for_difference_of_one,
            "1111011110": 0.0
        }
        assert (
            len(expected_bitstrings_to_differences)
            == len(actual_bitstrings_to_differences)
        ), "expected 4 * 4 results with valid bitstrings"
        assert (
            pytest.approx(expected_bitstrings_to_differences["1000010000"])
            == actual_bitstrings_to_differences["1000010000"]
        ), "expected only base energy for (0, 0)"
        assert (
            pytest.approx(expected_bitstrings_to_differences["1000011000"])
            == actual_bitstrings_to_differences["1000011000"]
        ), "expected base energy plus 0.5 * 1^2 * step^-2 for (0, 1)"
        assert (
            pytest.approx(expected_bitstrings_to_differences["1000011100"])
            == actual_bitstrings_to_differences["1000011100"]
        ), "expected base energy plus 0.5 * 2^2 * step^-2 for (0, 2)"
        assert (
            pytest.approx(expected_bitstrings_to_differences["1000011110"])
            == actual_bitstrings_to_differences["1000011110"]
        ), "expected base energy plus 0.5 * 3^2 * step^-2 for (0, 3)"
        assert (
            expected_bitstrings_to_differences
            == actual_bitstrings_to_differences
        ), "expected correct differences for all valid bitstrings"

    @pytest.mark.parametrize(
            "get_Hamiltonian, domain_wall_weighter, sampler_handler",
            [
                (
                    # The potential does not matter for the test of the kinetic
                    # term, so we just use the simpler case.
                    _get_spin_potential,
                    _spin_domain_wall_weighter,
                    _spin_sampler_handler
                ),
                (
                    _get_bit_potential,
                    _bit_domain_wall_weighter,
                    _bit_sampler_handler
                )
            ],
            ids=[
                "spin",
                "bit"
            ]
    )
    def test_all_valid_states_for_two_fields_in_zero_potential(
        self,
        *,
        get_Hamiltonian: Callable[[QftModelConfiguration], AnnealerHamiltonian],
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        """
        This tests that the exact solver finds the correct energies for each
        valid pair of ICDW configurations of two fields at neighboring radius
        values, removing any effect from the QFT potential by it being zero
        everywhere.
        """
        # We want the first field to take values 0, and 1 and the second to take
        # values 0, 1, and 2, leading to a maximum squared difference of
        # 1^2 + 2^2 = 5.
        number_of_first_field_values = 2
        number_of_second_field_values = 3
        first_field_definition = _get_test_field_definition(
            number_of_field_values=number_of_first_field_values,
            field_name="N"  # For normal as opposed to transpose
        )
        second_field_definition = _get_test_field_definition(
            number_of_field_values=number_of_second_field_values,
            field_name="T"  # For transpose as opposed to normal
        )
        potential_configuration = QftModelConfiguration(
            first_field=first_field_definition,
            second_field=second_field_definition,
            potential_in_quartic_GeV_per_field_step=[
                [0 for _ in range(number_of_first_field_values)]
                for _ in range(number_of_second_field_values)
            ]
        )

        annealer_Hamiltonian = get_Hamiltonian(potential_configuration)

        domain_end_fixing_weight, domain_wall_alignment_weight = (
            self._get_end_and_alignment_weights(annealer_Hamiltonian)
        )

        profiles_at_points = [
            FieldCollectionAtPoint(
                spatial_point_identifier=f"r{i}",
                spatial_radius_in_inverse_GeV=_radius_step_in_inverse_GeV,
                first_field=first_field_definition,
                second_field=second_field_definition
            ) for i in range(2)
        ]

        at_lower_radius = profiles_at_points[0]
        at_upper_radius = profiles_at_points[1]

        calculated_weights = domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=profiles_at_points,
            end_weight=domain_end_fixing_weight,
            alignment_weight=domain_wall_alignment_weight
        )
        calculated_weights.add(
            annealer_Hamiltonian.kinetic_weights(
                radius_step_in_inverse_GeV=_radius_step_in_inverse_GeV,
                nearer_center=at_lower_radius,
                nearer_edge=at_upper_radius,
                scaling_factor=1.0
            )
        )

        sample_provider = SampleProvider(
            sampler_name="exact",
            sampler_handler=sampler_handler
        )
        sampling_result = sample_provider.get_sample(calculated_weights)

        # We have the order (N at 0)(T at 0)(N at 1)(T at 1) for first field N,
        # second field T, at radius value 0 and radius value 1, respectively.
        actual_bitstrings_to_energies = (
            sample_provider.bitstrings_to_energies(
                binary_variable_names=(
                    at_lower_radius.first_field.binary_variable_names
                    + at_lower_radius.second_field.binary_variable_names
                    + at_upper_radius.first_field.binary_variable_names
                    + at_upper_radius.second_field.binary_variable_names
                ),
                sample_set=sampling_result
            )
        )

        # The lowest energy is going to be dependent on whether we use spin or
        # bit variables, so we just calculate differences from the easiest
        # state: 10010001001000 (N = 100 then T = 1000 at the lower radius then
        # the same again at the upper radius), which should have the lowest
        # energy anyway (jointly with other states of zero difference between
        # the fields).
        # We grab all the results under the energy which should only happen if
        # the alignments violate the conditions of the Ising-chain domain wall.
        actual_bitstrings_to_differences = (
            _as_differences_from_lowest_energy(
                bitstrings_to_absolute_energies=actual_bitstrings_to_energies,
                absolute_tolerance_in_quartic_GeV=(
                    0.75 * domain_wall_alignment_weight
                )
            )
        )

        field_step_squared = _field_step_in_GeV * _field_step_in_GeV
        radius_step_squared = (
            _radius_step_in_inverse_GeV * _radius_step_in_inverse_GeV
        )
        base_extra = (
            (0.5 * field_step_squared) / radius_step_squared
        )

        expected_bitstrings_to_differences = {
            # (0, 0) - (x, y)
            "10010001001000": (0.0 + 0.0) * base_extra,  # (0, 0) - (0, 0)
            "10010001001100": (0.0 + 1.0) * base_extra,  # (0, 0) - (0, 1)
            "10010001001110": (0.0 + 4.0) * base_extra,  # (0, 0) - (0, 2)
            "10010001101000": (1.0 + 0.0) * base_extra,  # (0, 0) - (1, 0)
            "10010001101100": (1.0 + 1.0) * base_extra,  # (0, 0) - (1, 1)
            "10010001101110": (1.0 + 4.0) * base_extra,  # (0, 0) - (1, 2)
            # (0, 1) - (x, y)
            "10011001001000": (0.0 + 1.0) * base_extra,  # (0, 1) - (0, 0)
            "10011001001100": (0.0 + 0.0) * base_extra,  # (0, 1) - (0, 1)
            "10011001001110": (0.0 + 1.0) * base_extra,  # (0, 1) - (0, 2)
            "10011001101000": (1.0 + 1.0) * base_extra,  # (0, 1) - (1, 0)
            "10011001101100": (1.0 + 0.0) * base_extra,  # (0, 1) - (1, 1)
            "10011001101110": (1.0 + 1.0) * base_extra,  # (0, 1) - (1, 2)
            # (0, 2) - (x, y)
            "10011101001000": (0.0 + 4.0) * base_extra,  # (0, 2) - (0, 0)
            "10011101001100": (0.0 + 1.0) * base_extra,  # (0, 2) - (0, 1)
            "10011101001110": (0.0 + 0.0) * base_extra,  # (0, 2) - (0, 2)
            "10011101101000": (1.0 + 4.0) * base_extra,  # (0, 2) - (1, 0)
            "10011101101100": (1.0 + 1.0) * base_extra,  # (0, 2) - (1, 1)
            "10011101101110": (1.0 + 0.0) * base_extra,  # (0, 2) - (1, 2)
            # (1, 0) - (x, y)
            "11010001001000": (1.0 + 0.0) * base_extra,  # (1, 0) - (0, 0)
            "11010001001100": (1.0 + 1.0) * base_extra,  # (1, 0) - (0, 1)
            "11010001001110": (1.0 + 4.0) * base_extra,  # (1, 0) - (0, 2)
            "11010001101000": (0.0 + 0.0) * base_extra,  # (1, 0) - (1, 0)
            "11010001101100": (0.0 + 1.0) * base_extra,  # (1, 0) - (1, 1)
            "11010001101110": (0.0 + 4.0) * base_extra,  # (1, 0) - (1, 2)
            # (1, 1) - (x, y)
            "11011001001000": (1.0 + 1.0) * base_extra,  # (1, 1) - (0, 0)
            "11011001001100": (1.0 + 0.0) * base_extra,  # (1, 1) - (0, 1)
            "11011001001110": (1.0 + 1.0) * base_extra,  # (1, 1) - (0, 2)
            "11011001101000": (0.0 + 1.0) * base_extra,  # (1, 1) - (1, 0)
            "11011001101100": (0.0 + 0.0) * base_extra,  # (1, 1) - (1, 1)
            "11011001101110": (0.0 + 1.0) * base_extra,  # (1, 1) - (1, 2)
            # (1, 2) - (x, y)
            "11011101001000": (1.0 + 4.0) * base_extra,  # (1, 2) - (0, 0)
            "11011101001100": (1.0 + 1.0) * base_extra,  # (1, 2) - (0, 1)
            "11011101001110": (1.0 + 0.0) * base_extra,  # (1, 2) - (0, 2)
            "11011101101000": (0.0 + 4.0) * base_extra,  # (1, 2) - (1, 0)
            "11011101101100": (0.0 + 1.0) * base_extra,  # (1, 2) - (1, 1)
            "11011101101110": (0.0 + 0.0) * base_extra,  # (1, 2) - (1, 2)
        }
        assert (
            len(expected_bitstrings_to_differences)
            == len(actual_bitstrings_to_differences)
        ), "expected (2*3)^2 results with valid bitstrings"
        # All the energies and differences of energies should be exactly
        # representable in binary so we can make floating-point number
        # comparisons without needing a tolerance.
        assert (
            expected_bitstrings_to_differences
            == actual_bitstrings_to_differences
        ), "expected correct differences for all valid bitstrings"

    @pytest.mark.parametrize(
            "get_Hamiltonian, domain_wall_weighter, sampler_handler",
            [
                (
                    _get_spin_potential,
                    _spin_domain_wall_weighter,
                    _spin_sampler_handler
                ),
                (
                    _get_bit_potential,
                    _bit_domain_wall_weighter,
                    _bit_sampler_handler
                )
            ],
            ids=[
                "spin",
                "bit"
            ]
    )
    def test_linear_potential_minimized_correctly_for_single_field(
        self,
        *,
        get_Hamiltonian: Callable[[QftModelConfiguration], AnnealerHamiltonian],
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        # We want a minimum at 10000000, hence 7 values.
        number_of_field_values = 7
        field_definition = _get_test_field_definition(
            number_of_field_values=number_of_field_values
        )
        potential_configuration = _get_single_field_potential_configuration(
            field_definition=field_definition,
            potential_function=_linear_single_field_potential
        )
        annealer_Hamiltonian = get_Hamiltonian(potential_configuration)

        all_bitstrings_to_energies = (
            self._get_all_bitstrings_to_energies_for_potential_at_single_point(
                annealer_Hamiltonian=annealer_Hamiltonian,
                domain_wall_weighter=domain_wall_weighter,
                sampler_handler=sampler_handler,
                first_field_definition=field_definition
            )
        )
        actual_bitstrings_to_differences = (
            _as_differences_from_lowest_energy(
                bitstrings_to_absolute_energies=all_bitstrings_to_energies,
                absolute_tolerance_in_quartic_GeV=1.0
            )
        )

        assert (
            1 == len(actual_bitstrings_to_differences)
        ), "expected only one minimum"
        actual_solution_bitstring = next(
            iter(actual_bitstrings_to_differences.keys())
        )
        assert (
            "10000000" == actual_solution_bitstring
        ), "expected domain wall completely to the left"

    @pytest.mark.parametrize(
            "get_Hamiltonian, domain_wall_weighter, sampler_handler",
            [
                (
                    _get_spin_potential,
                    _spin_domain_wall_weighter,
                    _spin_sampler_handler
                ),
                (
                    _get_bit_potential,
                    _bit_domain_wall_weighter,
                    _bit_sampler_handler
                )
            ],
            ids=[
                "spin",
                "bit"
            ]
    )
    def test_quadratic_potential_minimized_correctly_for_single_field(
        self,
        *,
        get_Hamiltonian: Callable[[QftModelConfiguration], AnnealerHamiltonian],
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        # We want a minimum at 11111100, hence 7 values.
        number_of_field_values = 7
        field_definition = _get_test_field_definition(
            number_of_field_values=number_of_field_values
        )
        potential_configuration = _get_single_field_potential_configuration(
            field_definition=field_definition,
            potential_function=_quadratic_single_field_potential
        )
        annealer_Hamiltonian = get_Hamiltonian(potential_configuration)

        all_bitstrings_to_energies = (
            self._get_all_bitstrings_to_energies_for_potential_at_single_point(
                annealer_Hamiltonian=annealer_Hamiltonian,
                domain_wall_weighter=domain_wall_weighter,
                sampler_handler=sampler_handler,
                first_field_definition=field_definition
            )
        )
        actual_bitstrings_to_differences = (
            _as_differences_from_lowest_energy(
                bitstrings_to_absolute_energies=all_bitstrings_to_energies,
                absolute_tolerance_in_quartic_GeV=1.0
            )
        )

        assert (
            1 == len(actual_bitstrings_to_differences)
        ), "expected only one minimum"
        actual_solution_bitstring = next(
            iter(actual_bitstrings_to_differences.keys())
        )
        assert (
            "11111100" == actual_solution_bitstring
        ), "expected 5 1s between fixed 1st bit and domain wall"

    @pytest.mark.parametrize(
            "get_Hamiltonian, domain_wall_weighter, sampler_handler",
            [
                (
                    _get_spin_potential,
                    _spin_domain_wall_weighter,
                    _spin_sampler_handler
                ),
                (
                    _get_bit_potential,
                    _bit_domain_wall_weighter,
                    _bit_sampler_handler
                )
            ],
            ids=[
                "spin",
                "bit"
            ]
    )
    def test_all_states_from_single_field_potential_present_and_correct(
        self,
        *,
        get_Hamiltonian: Callable[[QftModelConfiguration], AnnealerHamiltonian],
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        potential_values = [1.0, 2.0, -1.0, 3.0, -2.0, 4.0]
        number_of_values = len(potential_values)
        field_definition = _get_test_field_definition(
            number_of_field_values=number_of_values
        )
        potential_configuration = QftModelConfiguration(
            first_field=field_definition,
            potential_in_quartic_GeV_per_field_step=[potential_values]
        )
        annealer_Hamiltonian = get_Hamiltonian(potential_configuration)

        all_bitstrings_to_energies = (
            self._get_all_bitstrings_to_energies_for_potential_at_single_point(
                annealer_Hamiltonian=annealer_Hamiltonian,
                domain_wall_weighter=domain_wall_weighter,
                sampler_handler=sampler_handler,
                first_field_definition=field_definition
            )
        )
        actual_bitstrings_to_differences = (
            _as_differences_from_lowest_energy(
                bitstrings_to_absolute_energies=all_bitstrings_to_energies,
                absolute_tolerance_in_quartic_GeV=(
                    # We need to see a state at the maximum difference but we
                    # will ignore anything above that, since they should have
                    # invalid domain-wall configurations (and we check after
                    # this that we really did get all the valid states).
                    annealer_Hamiltonian.get_maximum_potential_difference()
                    + 1.0
                )
            )
        )

        # Since the state with the lowest energy should be "1111100" with -2.0,
        # we expect the potential values + 2.0 in each case.
        expected_bitstrings_to_differences = {
            "1000000": (potential_values[0] + 2.0),
            "1100000": (potential_values[1] + 2.0),
            "1110000": (potential_values[2] + 2.0),
            "1111000": (potential_values[3] + 2.0),
            "1111100": 0.0,
            "1111110": (potential_values[5] + 2.0)
        }

        # All the energies and differences of energies should be exactly
        # representable in binary so we can make floating-point number
        # comparisons without needing a tolerance.
        assert (
            len(expected_bitstrings_to_differences.keys())
            == len(actual_bitstrings_to_differences.keys())
        ), "incorrect number of valid states"
        assert (
            expected_bitstrings_to_differences.keys()
            == actual_bitstrings_to_differences.keys()
        ), "incorrect valid states"
        assert (
            expected_bitstrings_to_differences
            == actual_bitstrings_to_differences
        ), "incorrect energies for states"

    @pytest.mark.parametrize(
            "get_Hamiltonian, domain_wall_weighter, sampler_handler",
            [
                (
                    _get_spin_potential,
                    _spin_domain_wall_weighter,
                    _spin_sampler_handler
                ),
                (
                    _get_bit_potential,
                    _bit_domain_wall_weighter,
                    _bit_sampler_handler
                )
            ],
            ids=[
                "spin",
                "bit"
            ]
    )
    def test_all_states_from_two_field_potential_present_and_correct(
        self,
        *,
        get_Hamiltonian: Callable[[QftModelConfiguration], AnnealerHamiltonian],
        domain_wall_weighter: DomainWallWeighter,
        sampler_handler: SamplerHandler
    ):
        potential_values = [
            [2.0, 3.0],  # 1001000, 1101000
            [-5.0, -7.0],  # 1001100, 1101100
            [11.0, 13.0]  # 1001110, 1101110
        ]
        first_field_definition = _get_test_field_definition(
            number_of_field_values=len(potential_values[0]),
            field_name="N"  # For normal as opposed to transpose
        )
        second_field_definition = _get_test_field_definition(
            number_of_field_values=len(potential_values),
            field_name="T"  # For transpose as opposed to normal
        )
        potential_configuration = QftModelConfiguration(
            first_field=first_field_definition,
            second_field=second_field_definition,
            potential_in_quartic_GeV_per_field_step=potential_values
        )
        annealer_Hamiltonian = get_Hamiltonian(potential_configuration)

        all_bitstrings_to_energies = (
            self._get_all_bitstrings_to_energies_for_potential_at_single_point(
                annealer_Hamiltonian=annealer_Hamiltonian,
                domain_wall_weighter=domain_wall_weighter,
                sampler_handler=sampler_handler,
                first_field_definition=first_field_definition,
                second_field_definition=second_field_definition
            )
        )
        actual_bitstrings_to_differences = (
            _as_differences_from_lowest_energy(
                bitstrings_to_absolute_energies=all_bitstrings_to_energies,
                absolute_tolerance_in_quartic_GeV=(
                    # We need to see a state at the maximum difference but we
                    # will ignore anything above that, since they should have
                    # invalid domain-wall configurations (and we check after
                    # this that we really did get all the valid states).
                    annealer_Hamiltonian.get_maximum_potential_difference()
                    + 1.0
                )
            )
        )

        # Since the state with the lowest energy should be "1101100" with -7.0,
        # we expect the potential values + 7.0 in each case.
        expected_bitstrings_to_differences = {
            "1001000": (potential_values[0][0] + 7.0),
            "1101000": (potential_values[0][1] + 7.0),
            "1001100": (potential_values[1][0] + 7.0),
            "1101100": 0.0,
            "1001110": (potential_values[2][0] + 7.0),
            "1101110": (potential_values[2][1] + 7.0)
        }

        # All the energies and differences of energies should be exactly
        # representable in binary so we can make floating-point number
        # comparisons without needing a tolerance.
        assert (
            len(expected_bitstrings_to_differences.keys())
            == len(actual_bitstrings_to_differences.keys())
        ), "incorrect number of valid states"
        assert (
            expected_bitstrings_to_differences.keys()
            == actual_bitstrings_to_differences.keys()
        ), "incorrect valid states"
        assert (
            expected_bitstrings_to_differences
            == actual_bitstrings_to_differences
        ), "incorrect energies for states"

    def _get_end_and_alignment_weights(
            self,
            annealer_Hamiltonian: AnnealerHamiltonian
    ) -> tuple[float, float]:
        # This is copied from the internals of BubbleProfile.
        maximum_variable_weight = (
            annealer_Hamiltonian.get_maximum_potential_difference()
            + annealer_Hamiltonian.get_maximum_kinetic_contribution(
                _radius_step_in_inverse_GeV
            )
        )
        domain_wall_alignment_weight = 2.0 * maximum_variable_weight
        domain_end_fixing_weight = 2.0 * domain_wall_alignment_weight
        return (domain_end_fixing_weight, domain_wall_alignment_weight)

    def _get_all_bitstrings_to_energies_for_potential_at_single_point(
            self,
            *,
            annealer_Hamiltonian: AnnealerHamiltonian,
            domain_wall_weighter: DomainWallWeighter,
            sampler_handler: SamplerHandler,
            first_field_definition: FieldDefinition,
            second_field_definition: Optional[FieldDefinition] = None
    ) -> dict[str, float]:
        domain_end_fixing_weight, domain_wall_alignment_weight = (
            self._get_end_and_alignment_weights(annealer_Hamiltonian)
        )

        # One point is sufficient.
        profiles_at_points = [
            FieldCollectionAtPoint(
                spatial_point_identifier="r",
                spatial_radius_in_inverse_GeV=_radius_step_in_inverse_GeV,
                first_field=first_field_definition,
                second_field=second_field_definition
            )
        ]

        first_field = profiles_at_points[0].first_field
        second_field = profiles_at_points[0].second_field

        calculated_weights = domain_wall_weighter.weights_for_domain_walls(
            profiles_at_points=profiles_at_points,
            end_weight=domain_end_fixing_weight,
            alignment_weight=domain_wall_alignment_weight
        )
        calculated_weights.add(
            annealer_Hamiltonian.potential_weights(
                first_field=first_field,
                second_field=second_field,
                scaling_factor=1.0
            )
        )

        sample_provider = SampleProvider(
            sampler_name="exact",
            sampler_handler=sampler_handler
        )

        sampling_result = sample_provider.get_sample(calculated_weights)

        variable_names = (
            first_field.binary_variable_names
            if not second_field
            else (
                first_field.binary_variable_names
                + second_field.binary_variable_names
            )
        )

        return sample_provider.bitstrings_to_energies(
            binary_variable_names=variable_names,
            sample_set=sampling_result
        )
