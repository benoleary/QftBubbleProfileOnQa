from typing import Callable, Tuple
import xml.etree.ElementTree


def create_input():
    """
    This provides a means of setting up the input file for a simple case.
    """
    parameters_for_SM_Higgs = _for_SM_Higgs(
        linear_factor=0.01,
        number_of_steps_from_origin_to_VEV=2,
        number_of_spatial_steps=10
    )
    parameters_for_ACS = _for_ACS(
        N=50,
        M=50
    )
    (
        first_field_offset_in_GeV,
        first_field_step_in_GeV,
        spatial_step_in_inverse_GeV,
        potential_in_quartic_GeV,
        number_of_field_values,
        number_of_spatial_steps
    ) = parameters_for_ACS

    potential_in_quartic_GeV_per_field_step = [
        potential_in_quartic_GeV(f) for f in range(number_of_field_values)
    ]

    root_element = xml.etree.ElementTree.Element("configuration")
    _add(root_element, "sampler_name", "kerberos")
    _add(root_element, "output_CSV_filename", "example.csv")
    _add(root_element, "command_for_gnuplot", "/usr/bin/gnuplot")
    _add(root_element, "first_field_name", "f")
    _add(root_element, "number_of_spatial_steps", str(number_of_spatial_steps))
    _add(
        root_element,
        "first_field_offset_in_GeV",
        str(first_field_offset_in_GeV)
    )
    _add(root_element, "first_field_step_in_GeV", str(first_field_step_in_GeV))
    _add(
        root_element,
        "spatial_step_in_inverse_GeV",
        str(spatial_step_in_inverse_GeV)
    )
    _add(
        root_element,
        "potential_in_quartic_GeV_per_field_step",
        ";".join(str(v) for v in potential_in_quartic_GeV_per_field_step)
    )
    _add(root_element, "number_of_shots", "1000")
    xml.etree.ElementTree.ElementTree(root_element).write(
        "created_example.xml",
        encoding="utf8"
    )


def _for_SM_Higgs(
        *,
        linear_factor: float,
        number_of_steps_from_origin_to_VEV: int,
        number_of_spatial_steps: int
    ) -> Tuple[float, float, float, Callable[[int], float], int]:
    VEV_in_GeV = 246.0
    squared_VEV = VEV_in_GeV * VEV_in_GeV
    quartic_VEV = squared_VEV * squared_VEV
    mass_of_Higgs_in_GeV = 125.0
    squared_Higgs_mass = mass_of_Higgs_in_GeV * mass_of_Higgs_in_GeV
    lambda_coupling = squared_Higgs_mass / (2.0 * squared_VEV)
    step_factor = 1.0 / number_of_steps_from_origin_to_VEV

    def potential_in_quartic_GeV(f: int) -> float:
        h = (step_factor * f) - 1.0
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
        potential_in_quartic_GeV,
        (2 * number_of_steps_from_origin_to_VEV) + 1,
        number_of_spatial_steps
    )


def _for_ACS(
        *,
        N: int,
        M: int
    ) -> Tuple[float, float, float, Callable[[int], float]]:
    def potential_in_quartic_GeV(f: int) -> float:
        # V = (lambda/8) (phi^2 - a^2)^2 + (epsilon/2a)(phi - a)
        # a = lambda = 1, epsilon = 0.01
        # V = (1/8) (phi^2 - 1)^2 + (0.005)(phi - 1)
        # and phi is normalized to be between -1.0 and +1.0 inclusive.
        phi = ((2.0 * f) / N) - 1.0
        phi_squared_minus_one = (phi * phi) - 1.0
        return (
            (0.125 * phi_squared_minus_one * phi_squared_minus_one)
            + (0.000 * (phi - 1.0))
        )

    return (
        -1.0,
        # The values of phi have to go from -1 to +1.
        (2.0 / N),
        # A guess, because they plot rho from 0 to 25 for M = 50.
        0.5,
        potential_in_quartic_GeV,
        N + 1,
        M
    )


def _add(
        root_element: xml.etree.ElementTree.Element,
        element_name: str,
        element_text: str
    ):
    child_element = xml.etree.ElementTree.SubElement(root_element, element_name)
    child_element.text = element_text


if __name__ == '__main__':
    create_input()
