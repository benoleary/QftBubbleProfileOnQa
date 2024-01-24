import argparse

import minimization.sampling
import basis.variable
from input.configuration import FullConfiguration
from structure.bubble import BubbleProfile


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_file")
    parsed_arguments = argument_parser.parse_args()

    full_configuration = FullConfiguration(parsed_arguments.input_file)

    # TODO: fix from here
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
