import xml.etree.ElementTree


def create_input():
    """
    This provides a means of setting up the input file for a simple case.
    """
    root_element = xml.etree.ElementTree.Element("configuration")
    _add(root_element, "first_field_name", "f")
    _add(root_element, "number_of_spatial_steps", "10")

    VEV_in_GeV = 246.0
    squared_VEV = VEV_in_GeV * VEV_in_GeV
    quartic_VEV = squared_VEV * squared_VEV
    mass_of_Higgs_in_GeV = 125.0
    squared_Higgs_mass = mass_of_Higgs_in_GeV * mass_of_Higgs_in_GeV
    lambda_coupling = squared_Higgs_mass / (2.0 * squared_VEV)
    number_of_steps_from_origin_to_VEV = 5
    step_factor = 1.0 / number_of_steps_from_origin_to_VEV

    _add(root_element, "field_step_in_GeV", str(step_factor * VEV_in_GeV))

    # This is almost certainly completely wrong.
    _add(root_element, "spatial_step_in_inverse_GeV", str(0.5/VEV_in_GeV))

    dimensionless_linear_factor = 0.01
    def scaled_potential(f: int) -> float:
        h = (step_factor * f) - 1.0
        return (
            (0.25 * (h**4)) - (0.5 * (h**2)) + (dimensionless_linear_factor * h)
        )

    potential_in_quartic_GeV_per_field_step = [
        lambda_coupling * quartic_VEV * scaled_potential(f)
        for f in range((2 * number_of_steps_from_origin_to_VEV) + 1)
    ]
    _add(
        root_element,
        "potential_in_quartic_GeV_per_field_step",
        ";".join(str(v) for v in potential_in_quartic_GeV_per_field_step)
    )
    xml.etree.ElementTree.ElementTree(root_element).write("created_example.xml")


def _add(
        root_element: xml.etree.ElementTree.Element,
        element_name: str,
        element_text: str
    ):
    child_element = xml.etree.ElementTree.SubElement(root_element, element_name)
    child_element.text = element_text


if __name__ == '__main__':
    create_input()
