from typing import Callable, Tuple

def for_SM_Higgs(
        *,
        linear_factor: float,
        number_of_steps_from_origin_to_VEV: int,
        number_of_spatial_steps: int
    ) -> Tuple[
        float,
        float,
        Callable[[int], float],
        Callable[[float], float],
        int,
        int
    ]:
    VEV_in_GeV = 246.0
    squared_VEV = VEV_in_GeV * VEV_in_GeV
    quartic_VEV = squared_VEV * squared_VEV
    mass_of_Higgs_in_GeV = 125.0
    squared_Higgs_mass = mass_of_Higgs_in_GeV * mass_of_Higgs_in_GeV
    lambda_coupling = squared_Higgs_mass / (2.0 * squared_VEV)
    step_factor = 1.0 / number_of_steps_from_origin_to_VEV

    def field_to_GeV(f: int) -> float:
        return (step_factor * f) - 1.0

    def potential_in_quartic_GeV_from_field_in_GeV(h: float) -> float:
        return (
            lambda_coupling
            * quartic_VEV
            * ((0.25 * (h**4)) - (0.5 * (h**2)) + (linear_factor * h))
        )

    return (
        VEV_in_GeV,
        # This is almost certainly completely wrong.
        (0.5 / VEV_in_GeV),
        field_to_GeV,
        potential_in_quartic_GeV_from_field_in_GeV,
        (2 * number_of_steps_from_origin_to_VEV) + 1,
        number_of_spatial_steps
    )


def for_ACS(
        *,
        N: int,
        M: int
    ) -> Tuple[
        float,
        float,
        Callable[[int], float],
        Callable[[float], float],
        int,
        int
    ]:
    # V = (lambda/8) (phi^2 - a^2)^2 + (epsilon/2a)(phi - a)
    # a = lambda = 1, epsilon = 0.01
    # V = (1/8) (phi^2 - 1)^2 + (0.005)(phi - 1)
    # and phi is normalized to be between -1.0 and +1.0 inclusive.

    def field_to_GeV(f: int) -> float:
        return ((2.0 * f) / N) - 1.0

    def potential_in_quartic_GeV_from_field_in_GeV(phi: float) -> float:
        phi_squared_minus_one = (phi * phi) - 1.0
        return (
            (0.125 * phi_squared_minus_one * phi_squared_minus_one)
            + (0.01 * (phi - 1.0))
        )

    return (
        1.0,
        # A guess, because they plot rho from 0 to 25 for M = 50.
        0.5,
        field_to_GeV,
        potential_in_quartic_GeV_from_field_in_GeV,
        N + 1,
        M
    )
