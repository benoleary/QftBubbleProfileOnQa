import argparse
import xml.etree.ElementTree
from dimod import SimulatedAnnealingSampler
import minimization.sampling
import minimization.variable
from configuration.configuration import DiscreteConfiguration
from structure.bubble import BubbleProfile

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_file")
    argument_parser.add_argument("-l", "--local", action="store_true")
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
    bubble_profile = BubbleProfile(input_configuration)


    if parsed_arguments.local:
        message_for_Leap = None
        message_end = " locally with dimod.SimulatedAnnealingSampler"
    else:
        message_for_Leap = (
            f"QftBubbleProfileOnQa for input file {parsed_arguments.input_file}"
        )
        message_end = (
            " online via Leap with a dwave.system.EmbeddingComposite using a"
            " dwave.system.DWaveSampler"
        )
    print(
        f"About to run with configuration from {parsed_arguments.input_file}"
        + message_end
    )
    sample_set = minimization.sampling.get_sample(
        spin_biases=bubble_profile.spin_biases,
        message_for_Leap=message_for_Leap,
        number_of_shots=1000,
        local_sampler=SimulatedAnnealingSampler()
    )

    minimization.variable.print_bitstrings(
        "lowest energies:",
        sample_set.lowest(atol=bubble_profile.maximum_variable_weight)
    )


if __name__ == '__main__':
    main()
