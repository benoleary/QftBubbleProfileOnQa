from __future__ import annotations
import argparse
from collections.abc import Iterable
from typing import Optional
import xml.etree.ElementTree

import comparison.parameters


def create_input(
        *,
        model_name: str,
        has_second_field: bool
    ):
    """
    This provides a means of setting up the input file for a simple case.
    """
    if model_name == "sm":
        model_parameters = comparison.parameters.inspired_by_SM_Higgs(
            linear_factor=0.01,
            number_of_steps_from_origin_to_VEV=2,
            number_of_spatial_steps=10,
            has_second_field=has_second_field
        )
    elif model_name == "acs":
        model_parameters = comparison.parameters.for_ACS(
            N=20,
            M=20,
            epsilon=0.01,
            has_second_field=has_second_field
        )
    else:
        raise NotImplementedError(f"unknown model {model_name}")

    potential_in_quartic_GeV_per_field_step = [
        [
            model_parameters.potential_in_quartic_GeV_from_fields_in_GeV(
                model_parameters.first_field_to_GeV(f),
                model_parameters.second_field_to_GeV(s)
            )
            for f in range(model_parameters.number_of_first_field_values)
        ] for s in range(model_parameters.number_of_second_field_values or 1)
    ]

    root_element = xml.etree.ElementTree.Element("configuration")
    _add_qft_element(
        root_element=root_element,
        first_field_bound_in_GeV=model_parameters.first_field_bound_in_GeV,
        second_field_bound_in_GeV=model_parameters.second_field_bound_in_GeV,
        potential_in_quartic_GeV_per_field_step=(
            potential_in_quartic_GeV_per_field_step
        )
    )
    _add_space_element(
        root_element=root_element,
        number_of_spatial_steps=model_parameters.number_of_spatial_steps,
        spatial_step_in_inverse_GeV=model_parameters.spatial_step_in_inverse_GeV
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
        second_field_bound_in_GeV: Optional[float],
        potential_in_quartic_GeV_per_field_step: Iterable[Iterable[float]]
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
    if second_field_bound_in_GeV:
        second_field_element = xml.etree.ElementTree.SubElement(
            qft_element,
            "second_field"
        )
        _add(second_field_element, "field_name", "g")
        _add(
            second_field_element,
            "lower_bound_in_GeV",
            "0.0"
        )
        _add(
            second_field_element,
            "upper_bound_in_GeV",
            str(second_field_bound_in_GeV)
        )
        _add(
            second_field_element,
            "true_vacuum_value_in_GeV",
            "0.0"
        )
        _add(
            second_field_element,
            "false_vacuum_value_in_GeV",
            "0.0"
        )
    _add(
        qft_element,
        "potential_in_quartic_GeV_per_field_step",
        "#".join([
            ";".join(str(v) for v in constant_second)
            for constant_second in potential_in_quartic_GeV_per_field_step
        ])
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
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_name")
    argument_parser.add_argument("--second_field", action="store_true")
    parsed_arguments = argument_parser.parse_args()

    create_input(
        model_name=parsed_arguments.model_name,
        has_second_field=parsed_arguments.second_field
    )
