from typing import Optional
import argparse
import xml.etree.ElementTree
import minimization.sampling
import basis.variable
from input.configuration import DiscreteConfiguration, FieldDefinition
from structure.bubble import BubbleProfile


def xml_str(
        *,
        parent_element: xml.etree.ElementTree,
        element_name: str
    ) -> Optional[str]:
        xml_element = parent_element.find(element_name)
        # The elements are not truthy in an intuitive way! We have to check
        # against None.
        if xml_element is None:
            return None
        return xml_element.text

def xml_int(
        *,
        parent_element: xml.etree.ElementTree,
        element_name: str
    ) -> Optional[int]:
    element_text = xml_str(
        parent_element=parent_element,
        element_name=element_name
    )
    return int(element_text) if element_text else None

def xml_float(
        *,
        parent_element: xml.etree.ElementTree,
        element_name: str
    ) -> Optional[float]:
    element_text = xml_str(
        parent_element=parent_element,
        element_name=element_name
    )
    return float(element_text) if element_text else None


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_file")
    parsed_arguments = argument_parser.parse_args()

    input_xml_root = (
        xml.etree.ElementTree.parse(parsed_arguments.input_file).getroot()
    )

    def root_xml_str(element_name: str) -> Optional[str]:
        return xml_str(
            parent_element=input_xml_root,
            element_name=element_name
        )
    def root_xml_int(element_name: str) -> Optional[int]:
        return xml_int(
            parent_element=input_xml_root,
            element_name=element_name
        )
    def root_xml_float(element_name: str) -> Optional[float]:
        return xml_float(
            parent_element=input_xml_root,
            element_name=element_name
        )

    potential_element = input_xml_root.find(
        "potential_in_quartic_GeV_per_field_step"
    )
    if potential_element is None:
        raise ValueError("No XML element for potential")
    potential_per_field_step=[
        [float(v) for v in potential_row.split(";")]
        for potential_row in potential_element.text.split("#")
    ]
    print(f"potential_per_field_step = {potential_per_field_step}")

    def root_xml_field_definition(
            *,
            element_name: str,
            number_of_values: int
        ) -> Optional[float]:
        field_element = input_xml_root.find(element_name)
        # The elements are not truthy in an intuitive way! We have to check
        # against None.
        if field_element is None:
            return None
        return FieldDefinition(
                field_name=xml_str(
                    parent_element=field_element,
                    element_name="field_name"
                ),
                number_of_values=number_of_values,
                lower_bound_in_GeV=xml_float(
                    parent_element=field_element,
                    element_name="lower_bound_in_GeV"
                ),
                upper_bound_in_GeV=xml_float(
                    parent_element=field_element,
                    element_name="upper_bound_in_GeV"
                ),
                true_vacuum_value_in_GeV=xml_float(
                    parent_element=field_element,
                    element_name="true_vacuum_value_in_GeV"
                ),
                false_vacuum_value_in_GeV=xml_float(
                    parent_element=field_element,
                    element_name="false_vacuum_value_in_GeV"
                )
            )

    first_field = root_xml_field_definition(
        element_name="first_field",
        number_of_values=len(potential_per_field_step[0])
    )
    if first_field is None:
        raise ValueError("No XML element for first field")

    second_field = root_xml_field_definition(
        element_name="second_field",
        number_of_values=len(potential_per_field_step)
    )

    input_configuration = DiscreteConfiguration(
        number_of_spatial_steps=root_xml_int("number_of_spatial_steps"),
        spatial_step_in_inverse_GeV=root_xml_float(
            "spatial_step_in_inverse_GeV"
        ),
        volume_exponent=root_xml_int("volume_exponent"),
        first_field=first_field,
        second_field=second_field,
        potential_in_quartic_GeV_per_field_step=potential_per_field_step,
        sampler_name=root_xml_str("sampler_name"),
        number_of_shots=root_xml_int("number_of_shots"),
        output_CSV_filename=root_xml_str("output_CSV_filename"),
        command_for_gnuplot=root_xml_str("command_for_gnuplot")
    )
    bubble_profile = BubbleProfile(input_configuration)

    message_for_Leap = (
        f"QftBubbleProfileOnQa for input file {parsed_arguments.input_file}"
    ) if input_configuration.sampler_name == "dwave" else None

    message_end = (
        f" online via Leap ({input_configuration.sampler_name})"
        if input_configuration.sampler_name in ("dwave", "kerberos")
        else f" locally ({input_configuration.sampler_name})"
    )

    print(
        f"About to run with configuration from {parsed_arguments.input_file}"
        + message_end
    )
    sample_set = minimization.sampling.get_sample(
        spin_biases=bubble_profile.spin_biases,
        message_for_Leap=message_for_Leap,
        number_of_shots=input_configuration.number_of_shots or 1000,
        sampler_name=input_configuration.sampler_name
    )

    basis.variable.print_bitstrings(
        "lowest energies:",
        sample_set.lowest(atol=bubble_profile.maximum_variable_weight)
    )

    converted_to_GeV = (
        bubble_profile.map_radius_labels_to_field_strengths_from_lowest_sample(
            sample_set
        )
    )
    print(converted_to_GeV)

    must_generate_CSV = bool(
        input_configuration.output_CSV_filename
        or input_configuration.command_for_gnuplot
    )
    if must_generate_CSV:
        content_for_CSV = bubble_profile.lowest_sample_as_CSV_file_content(
            sample_set
        )
        data_filename = (
            input_configuration.output_CSV_filename
            or "temporary_gnuplot_input.csv"
        )
        print(f"writing profile in {data_filename}")
        with open(data_filename, "w") as output_file:
            output_file.write("\n".join(content_for_CSV) + "\n")

        if input_configuration.command_for_gnuplot:
            picture_filename = data_filename.rsplit(".", 1)[0] + ".png"
            plotting_filename = "temporary_gnuplot_input.in"
            print(
                f"running {input_configuration.command_for_gnuplot} "
                f"on {plotting_filename}"
            )
            title_text = "Single field approximation, " + (
                f"thin-wall approximation"
                if input_configuration.volume_exponent == 0
                else f"volume exponent {input_configuration.volume_exponent}"
            )
            with open(plotting_filename, "w") as output_file:
                output_file.write(
                    f"set title \"{title_text}\"\n"
                    "set datafile separator \";\"\n"
                    "set key autotitle columnhead\n"
                    "unset key\n"
                    "set xlabel \"r in 1/GeV\"\n"
                    "set ylabel \"f in GeV\"\n"
                    "set term png\n"
                    f"set output \"{picture_filename}\"\n"
                    f"plot \"{data_filename}\""
                )
            import subprocess
            subprocess.call(
                f"{input_configuration.command_for_gnuplot}"
                f" {plotting_filename}",
                shell=True
            )


if __name__ == "__main__":
    main()
