from typing import Callable, Tuple

def for_SM_Higgs(
        *,
        linear_factor: float,
        number_of_steps_from_origin_to_VEV: int,
        number_of_spatial_steps: int
    ) -> Tuple[
        float,
        float,
        float,
        Callable[[int], float],
        Callable[[float], float],
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
        -VEV_in_GeV,
        (step_factor * VEV_in_GeV),
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
    ) -> Tuple[float, float, float, Callable[[int], float]]:
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
        -1.0,
        # The values of phi have to go from -1 to +1.
        (2.0 / N),
        # A guess, because they plot rho from 0 to 25 for M = 50.
        0.5,
        field_to_GeV,
        potential_in_quartic_GeV_from_field_in_GeV,
        N + 1,
        M
    )


def for_thick_ACS(
        *,
        N: int,
        M: int
    ) -> Tuple[float, float, float, Callable[[int], float]]:
    # want +- 1
    # (dV/df)/c = (x^2 - 1)(x - b) = x^3 - b x^2 - x + b
    # V/c = x^4/4 - bx^3/3 - x^2/2 + bx
    # diff = c ((1/4 - b/3 - 1/2 + b) - (1/4 + b/3 - 1/2 - b))
    # = -2 c b (1/3 - 1) = 4 b c / 3
    # -1 < b < 1 so need to use c
    # barrier height out of false
    # = c ((b^4/4 - b^4/3 - b^2/2 + b^2) - (1/4 - b/3 - 1/2 + b))
    # = c ((-b^4/12 + b^2/2) - (2b/3 - 1/4))
    # = c (-b^4/12 + b^2/2 - 2b/3 + 1/4)
    # b = 1/2, barrier/c = 1/4 - 1/3 + 1/8 - 1/198 = a bit under 1/24,
    # diff/c = 2/3
    # c = 1/2 so that quartic factor is 1/8, subtract 1/24 to have 0 at x = +1
    # V = x^4/8 - x^3/12 - x^2/4 + x/4 - 1/24
    # dV/dx = x^3/2 - x^2/4 - x/2 + 1/4

    def field_to_GeV(f: int) -> float:
        return ((2.0 * f) / N) - 1.0

    def potential_in_quartic_GeV_from_field_in_GeV(phi: float) -> float:
        return (
            (0.125 * (phi**4))
            - ((phi**3) / 12.0)
            - (0.25 * (phi**2))
            + (0.25 * phi)
            - (1.0 / 24.0)
        )

    return (
        -1.0,
        # The values of phi have to go from -1 to +1.
        (2.0 / N),
        0.5,
        field_to_GeV,
        potential_in_quartic_GeV_from_field_in_GeV,
        N + 1,
        M
    )
