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
    parameters_for_ACS = comparison.parameters.for_thick_ACS(
        N=50,
        M=50
    )
    (
        first_field_offset_in_GeV,
        first_field_step_in_GeV,
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
    _add(root_element, "volume_exponent", "0")
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


def _add(
        root_element: xml.etree.ElementTree.Element,
        element_name: str,
        element_text: str
    ):
    child_element = xml.etree.ElementTree.SubElement(root_element, element_name)
    child_element.text = element_text


if __name__ == '__main__':
    create_input()
