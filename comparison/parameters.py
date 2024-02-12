from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional


# We would like to use kw_only=True, but that needs Python 3.10 or later.
@dataclass(frozen=True, repr=False, eq=False)
class ExampleModelParameters:
    first_field_bound_in_GeV: float
    second_field_bound_in_GeV: Optional[float]
    first_field_to_GeV: Callable[[int], float]
    second_field_to_GeV: Callable[[int], float]
    potential_in_quartic_GeV_from_fields_in_GeV: Callable[[float, float], float]
    number_of_first_field_values: int
    number_of_second_field_values: Optional[int]
    spatial_step_in_inverse_GeV: float
    number_of_spatial_steps: int


def inspired_by_SM_Higgs(
        *,
        linear_factor: float,
        number_of_steps_from_origin_to_VEV: int,
        number_of_spatial_steps: int,
        has_second_field: bool
) -> ExampleModelParameters:
    VEV_in_GeV = 246.0
    squared_VEV = VEV_in_GeV * VEV_in_GeV
    cubic_VeV = VEV_in_GeV * squared_VEV
    mass_of_Higgs_in_GeV = 125.0
    squared_Higgs_mass = mass_of_Higgs_in_GeV * mass_of_Higgs_in_GeV
    lambda_coupling = squared_Higgs_mass / (2.0 * squared_VEV)
    step_factor = 1.0 / number_of_steps_from_origin_to_VEV

    def first_field_to_GeV(f: int) -> float:
        return ((step_factor * f) - 1.0) * VEV_in_GeV

    if not has_second_field:
        def potential_in_quartic_GeV_from_field_in_GeV(
                h: float,
                g: float
        ) -> float:
            return (
                lambda_coupling
                * (
                    (0.25 * (h**4))
                    - (0.5 * squared_VEV * (h**2))
                    + (linear_factor * cubic_VeV * h)
                )
            )

        return ExampleModelParameters(
            first_field_bound_in_GeV=VEV_in_GeV,
            second_field_bound_in_GeV=None,
            first_field_to_GeV=first_field_to_GeV,
            second_field_to_GeV=field_to_zero,
            potential_in_quartic_GeV_from_fields_in_GeV=(
                potential_in_quartic_GeV_from_field_in_GeV
            ),
            number_of_first_field_values=(
                (2 * number_of_steps_from_origin_to_VEV) + 1
            ),
            number_of_second_field_values=None,
            # This is almost certainly completely wrong.
            spatial_step_in_inverse_GeV=(0.5 / VEV_in_GeV),
            number_of_spatial_steps=number_of_spatial_steps
        )


    # The depth of the single field SM potential without the linear term is
    # a quarter of the quartic coupling times the fourth power of the VEV.
    depth_of_SM_potential = 0.25 * lambda_coupling * squared_VEV * squared_VEV

    # We can control the depth and position of the lowest point of the barrier
    # with a quartic term plus linear term.
    depth_at_lowest_barrier = 0.8 * depth_of_SM_potential
    second_strength_at_lowest_barrier = 0.1 * VEV_in_GeV
    second_linear = (
        (4.0 * depth_at_lowest_barrier)
        / (3.0 * second_strength_at_lowest_barrier)
    )
    second_quartic = (
        depth_at_lowest_barrier
        / (3.0 * (second_strength_at_lowest_barrier**4))
    )

    def second_field_to_GeV(g: int) -> float:
        return (step_factor * g * second_strength_at_lowest_barrier)

    def potential_in_quartic_GeV_from_field_in_GeV(h: float, g: float) -> float:
        hh = (h**2)
        purely_first = (
            lambda_coupling
            * (
                (0.25 * hh * hh)
                - (0.5 * squared_VEV * hh)
                + (linear_factor * cubic_VeV * h)
            )
        )
        gg = (g**2)
        purely_second = (second_quartic * gg * gg) - (second_linear * g)
        return (
            purely_first
            + (16.0 * hh * gg)
            + purely_second
        )

    return ExampleModelParameters(
        first_field_bound_in_GeV=VEV_in_GeV,
        second_field_bound_in_GeV=second_strength_at_lowest_barrier,
        first_field_to_GeV=first_field_to_GeV,
        second_field_to_GeV=second_field_to_GeV,
        potential_in_quartic_GeV_from_fields_in_GeV=(
            potential_in_quartic_GeV_from_field_in_GeV
        ),
        number_of_first_field_values=(
            (2 * number_of_steps_from_origin_to_VEV) + 1
        ),
        number_of_second_field_values=(number_of_steps_from_origin_to_VEV + 1),
        # This is almost certainly completely wrong.
        spatial_step_in_inverse_GeV=(0.5 / VEV_in_GeV),
        number_of_spatial_steps=number_of_spatial_steps
    )


def for_ACS(
        *,
        N: int,
        M: int,
        epsilon: float,
        has_second_field: bool
) -> ExampleModelParameters:
    # V = (lambda/8) (phi^2 - a^2)^2 + (epsilon/2a)(phi - a)
    # a = lambda = 1, epsilon = 0.01
    # V = (1/8) (phi^2 - 1)^2 + (0.005)(phi - 1)
    # and phi is normalized to be between -1.0 and +1.0 inclusive.

    def first_field_to_GeV(phi: int) -> float:
        return ((2.0 * phi) / N) - 1.0

    if not has_second_field:
        def potential_in_quartic_GeV_from_field_in_GeV(
                phi: float,
                psi: float
        ) -> float:
            phi_squared_minus_one = (phi * phi) - 1.0
            return (
                (0.125 * phi_squared_minus_one * phi_squared_minus_one)
                + (epsilon * (phi - 1.0))
            )

        return ExampleModelParameters(
            first_field_bound_in_GeV=1.0,
            second_field_bound_in_GeV=None,
            first_field_to_GeV=first_field_to_GeV,
            second_field_to_GeV=field_to_zero,
            potential_in_quartic_GeV_from_fields_in_GeV=(
                potential_in_quartic_GeV_from_field_in_GeV
            ),
            number_of_first_field_values=(N + 1),
            number_of_second_field_values=None,
            # A guess, because they plot rho from 0 to 25 for M = 50.
            spatial_step_in_inverse_GeV=0.5,
            number_of_spatial_steps=M
        )

    def second_field_to_GeV(psi: int) -> float:
        return ((2.0 * psi) / N)

    def potential_in_quartic_GeV_from_field_in_GeV(
            phi: float,
            psi: float
    ) -> float:
        phi_squared = (phi * phi)
        phi_squared_minus_one = phi_squared - 1.0
        psi_squared = (psi * psi)
        psi_squared_minus_scaling = (
            psi_squared - 0.9
        )
        return (
            (0.125 * phi_squared_minus_one * phi_squared_minus_one)
            + (0.125 * psi_squared_minus_scaling * psi_squared_minus_scaling)
            + (0.25 * psi_squared * psi_squared)
            + (epsilon * (phi - 1.0))
        )

    return ExampleModelParameters(
        first_field_bound_in_GeV=1.0,
        second_field_bound_in_GeV=1.0,
        first_field_to_GeV=first_field_to_GeV,
        second_field_to_GeV=second_field_to_GeV,
        potential_in_quartic_GeV_from_fields_in_GeV=(
            potential_in_quartic_GeV_from_field_in_GeV
        ),
        number_of_first_field_values=(N + 1),
        number_of_second_field_values=((N + 1) / 2),
        # A guess, because they plot rho from 0 to 25 for M = 50.
        spatial_step_in_inverse_GeV=0.5,
        number_of_spatial_steps=M
    )


def field_to_zero(x: int) -> float:
    return 0.0
