import argparse
import xml.etree.ElementTree
from configuration.configuration import DiscreteConfiguration

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_file")
    parsed_arguments = argument_parser.parse_args()

    input_xml_root = (
        xml.etree.ElementTree.parse(parsed_arguments.input_file).getroot()
    )

    potential_string = input_xml_root.find(
        "potential_in_quartic_GeV_per_field_step"
    ).text
    input_configuration = DiscreteConfiguration(
        first_field_name=input_xml_root.find("first_field_name").text,
        number_of_spatial_steps=int(
            input_xml_root.find("number_of_spatial_steps").text
        ),
        spatial_step_in_inverse_GeV=float(
            input_xml_root.find("spatial_step_in_inverse_GeV").text
        ),
        field_step_in_GeV=float(input_xml_root.find("field_step_in_GeV").text),
        potential_in_quartic_GeV_per_field_step=[
            float(v) for v in potential_string.split(";")
        ]
    )

    print(
        "parsed potential"
        f" {input_configuration.potential_in_quartic_GeV_per_field_step}"
    )

    # TODO: implement some functionality...
    print(
        "Imagine something wonderful happening, involving input taken from"
        f" {parsed_arguments.input_file}"
    )

if __name__ == '__main__':
    main()
