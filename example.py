from typing import List
import xml.etree.ElementTree
import comparison.parameters


def create_input():
    """
    This provides a means of setting up the input file for a simple case.
    """
    parameters_for_SM_Higgs = comparison.parameters.for_SM_Higgs(
        linear_factor=0.01,
        number_of_steps_from_origin_to_VEV=2,
        number_of_spatial_steps=10
    )
    parameters_for_ACS = comparison.parameters.for_ACS(
        N=50,
        M=50
    )
    (
        first_field_bound_in_GeV,
        spatial_step_in_inverse_GeV,
        field_to_GeV,
        potential_in_quartic_GeV_from_field_in_GeV,
        number_of_field_values,
        number_of_spatial_steps
    ) = parameters_for_ACS

    potential_in_quartic_GeV_per_field_step = [
        potential_in_quartic_GeV_from_field_in_GeV(field_to_GeV(f))
        for f in range(number_of_field_values)
    ]

    root_element = xml.etree.ElementTree.Element("configuration")
    _add_qft_element(
        root_element=root_element,
        first_field_bound_in_GeV=first_field_bound_in_GeV,
        potential_in_quartic_GeV_per_field_step=(
            potential_in_quartic_GeV_per_field_step
        )
    )
    _add_space_element(
        root_element=root_element,
        number_of_spatial_steps=number_of_spatial_steps,
        spatial_step_in_inverse_GeV=spatial_step_in_inverse_GeV
    )
    _add_annealer_element(root_element)
    _add_output_element(root_element)

    xml.etree.ElementTree.ElementTree(root_element).write(
        "example.xml",
        encoding="utf8"
    )


def _add(
        root_element: xml.etree.ElementTree.Element,
        element_name: str,
        element_text: str
    ):
    child_element = xml.etree.ElementTree.SubElement(root_element, element_name)
    child_element.text = element_text


def _add_qft_element(
        *,
        root_element: xml.etree.ElementTree.Element,
        first_field_bound_in_GeV: float,
        potential_in_quartic_GeV_per_field_step: List[float]
):
    qft_element = xml.etree.ElementTree.SubElement(
        root_element,
        "qft"
    )
    first_field_element = xml.etree.ElementTree.SubElement(
        qft_element,
        "first_field"
    )
    _add(first_field_element, "field_name", "f")
    _add(
        first_field_element,
        "lower_bound_in_GeV",
        str(-first_field_bound_in_GeV)
    )
    _add(
        first_field_element,
        "upper_bound_in_GeV",
        str(first_field_bound_in_GeV)
    )
    _add(
        first_field_element,
        "true_vacuum_value_in_GeV",
        str(-first_field_bound_in_GeV)
    )
    _add(
        first_field_element,
        "false_vacuum_value_in_GeV",
        str(first_field_bound_in_GeV)
    )
    _add(
        qft_element,
        "potential_in_quartic_GeV_per_field_step",
        ";".join(str(v) for v in potential_in_quartic_GeV_per_field_step)
    )


def _add_space_element(
        *,
        root_element: xml.etree.ElementTree.Element,
        number_of_spatial_steps: int,
        spatial_step_in_inverse_GeV: float
):
    space_element = xml.etree.ElementTree.SubElement(root_element, "space")
    _add(space_element, "number_of_spatial_steps", str(number_of_spatial_steps))
    _add(
        space_element,
        "spatial_step_in_inverse_GeV",
        str(spatial_step_in_inverse_GeV)
    )
    _add(space_element, "volume_exponent", "0")


def _add_annealer_element(root_element: xml.etree.ElementTree.Element):
    annealer_element = xml.etree.ElementTree.SubElement(
        root_element,
        "annealer"
    )
    _add(annealer_element, "sampler_name", "kerberos")
    _add(annealer_element, "variable_type", "spin")


def _add_output_element(root_element: xml.etree.ElementTree.Element):
    output_element = xml.etree.ElementTree.SubElement(root_element, "output")
    _add(output_element, "output_CSV_filename", "example.csv")
    _add(output_element, "command_for_gnuplot", "/usr/bin/gnuplot")


if __name__ == '__main__':
    create_input()
