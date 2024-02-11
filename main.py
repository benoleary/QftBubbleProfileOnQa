import argparse
from dataclasses import dataclass

from dimod import SampleSet

from basis.field import FieldDefinition
from dynamics.hamiltonian import AnnealerHamiltonian
from dynamics.bit import BitHamiltonian
from dynamics.spin import SpinHamiltonian
from input.configuration import QftModelConfiguration, FullConfiguration
from minimization.sampling import SampleProvider, SamplerHandler
from minimization.bit import BitSamplerHandler
from minimization.spin import SpinSamplerHandler
from output.printing import CsvWriter
from structure.bubble import BubbleProfile
from structure.domain_wall import DomainWallWeighter
from structure.bit import BitDomainWallWeighter
from structure.spin import SpinDomainWallWeighter


# We would like to use kw_only=True, but that needs Python 3.10 or later.
@dataclass(frozen=True, repr=False, eq=False)
class VariableTypeDependence:
    annealer_Hamiltonian: AnnealerHamiltonian
    domain_wall_weighter: DomainWallWeighter
    sample_handler: SamplerHandler


def get_variable_type_dependence(
        *,
        variable_type: str,
        QFT_model_configuration: QftModelConfiguration
    ) -> VariableTypeDependence:
    if variable_type == "spin":
        return VariableTypeDependence(
            annealer_Hamiltonian=SpinHamiltonian(QFT_model_configuration),
            domain_wall_weighter=SpinDomainWallWeighter(),
            sample_handler=SpinSamplerHandler()
        )
    if variable_type == "bit":
        return VariableTypeDependence(
            annealer_Hamiltonian=BitHamiltonian(QFT_model_configuration),
            domain_wall_weighter=BitDomainWallWeighter(),
            sample_handler=BitSamplerHandler()
        )
    raise ValueError(
        f"Unknown variable type \"{variable_type}\", allowed: \"spin\", \"bit\""
    )


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input_file")
    parsed_arguments = argument_parser.parse_args()

    full_configuration = FullConfiguration(parsed_arguments.input_file)
    variable_type_dependence = get_variable_type_dependence(
        variable_type=full_configuration.annealer_configuration.variable_type,
        QFT_model_configuration=full_configuration.QFT_model_configuration
    )

    bubble_profile = BubbleProfile(
        annealer_Hamiltonian=variable_type_dependence.annealer_Hamiltonian,
        domain_wall_weighter=variable_type_dependence.domain_wall_weighter,
        spatial_lattice_configuration=(
            full_configuration.spatial_lattice_configuration
        )
    )

    sampler_name = full_configuration.annealer_configuration.sampler_name
    message_for_Leap = (
        f"QftBubbleProfileOnQa for input file {parsed_arguments.input_file}"
    ) if sampler_name == "dwave" else None
    sample_provider = SampleProvider(
            sampler_name=sampler_name,
            sampler_handler=variable_type_dependence.sample_handler,
            message_for_Leap=message_for_Leap,
            number_of_shots=(
                full_configuration.annealer_configuration.number_of_shots
            )
    )
    message_end = (
        f" online via Leap ({sampler_name})"
        if sampler_name in ("dwave", "kerberos")
        else f" locally ({sampler_name})"
    )

    print(
        f"About to run with configuration from {parsed_arguments.input_file}"
        + message_end
    )
    sample_set = sample_provider.get_sample(bubble_profile.annealing_weights)

    sample_provider.print_bitstrings(
        title_message="lowest energies:",
        sample_set=sample_set.lowest(
            atol=bubble_profile.maximum_variable_weight
        )
    )

    lowest_energy_sample = sample_provider.get_lowest_from_set(sample_set)

    profile_points = bubble_profile.field_strengths_at_radius_values(
        solution_sample=lowest_energy_sample,
        sample_provider=sample_provider
    )
    converted_to_GeV = [
        f"r={p.radius_in_inverse_GeV} ({f.spatial_point_identifier}):"
        f" {f.first_field.field_definition.field_name}="
        f"{p.first_field_strength_in_GeV}"
        + (
            "" if not f.second_field
            else (
                f", {f.second_field.field_definition.field_name}="
                f"{p.second_field_strength_in_GeV}"
            )
        )
        for f, p in zip(bubble_profile.fields_at_points, profile_points)
    ]
    print(converted_to_GeV)

    # If there is a name for an output CSV file, use it; if not but we are
    # plotting with gnuplot, set a temporary filename, otherwise use None.
    output_CSV_filename = (
        full_configuration.output_configuration.output_CSV_filename
        or (
            "temporary_gnuplot_input.csv"
            if full_configuration.output_configuration.command_for_gnuplot
            else None
        )
    )

    if output_CSV_filename:
        output_filename_root = output_CSV_filename.rsplit(".", 1)[0]
        CSV_writer = CsvWriter(bubble_profile=bubble_profile)
        print(f"writing profile in {output_CSV_filename}")
        CSV_writer.write_file_from_sample(
            output_CSV_filename=output_CSV_filename,
            solution_sample=lowest_energy_sample,
            sample_provider=sample_provider
        )
        command_for_gnuplot = (
                full_configuration.output_configuration.command_for_gnuplot
            )
        if command_for_gnuplot:
            plot_bubble_profile(
                full_configuration=full_configuration,
                output_filename_root=output_filename_root,
                output_CSV_filename=output_CSV_filename,
                command_for_gnuplot=command_for_gnuplot
            )
            if not full_configuration.QFT_model_configuration.second_field:
                plot_potential_for_single_field(
                    QFT_model_configuration=(
                        full_configuration.QFT_model_configuration
                    ),
                    CSV_writer=CSV_writer,
                    output_filename_root=output_filename_root,
                    command_for_gnuplot=command_for_gnuplot
            )


def plot_bubble_profile(
        *,
        full_configuration: FullConfiguration,
        output_filename_root: str,
        output_CSV_filename: str,
        command_for_gnuplot: str
):
    picture_filename = output_filename_root + ".png"
    plotting_filename = "temporary_gnuplot_bubble_profile.in"
    print(f"running {command_for_gnuplot} on {plotting_filename}")
    volume_exponent = (
        full_configuration.spatial_lattice_configuration.volume_exponent
    )
    title_text = (
        "Single field approximation, "
        + (
            f"thin-wall approximation"
            if volume_exponent == 0
            else f"volume exponent {volume_exponent}"
        )
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
                    f"plot \"{output_CSV_filename}\""
                )
    import subprocess
    subprocess.call(
                f"{command_for_gnuplot} {plotting_filename}",
                shell=True
            )


def plot_potential_for_single_field(
        *,
        QFT_model_configuration: QftModelConfiguration,
        CSV_writer: CsvWriter,
        output_filename_root: str,
        command_for_gnuplot: str
):
    data_filename = output_filename_root + "_potential.csv"
    picture_filename = output_filename_root + "_potential.png"
    plotting_filename = "temporary_gnuplot_potential.in"
    plotted_field = QFT_model_configuration.first_field
    field_name = plotted_field.field_name
    print(f"running {command_for_gnuplot} on {plotting_filename}")

    def field_in_GeV(step_index: int) -> float:
        return (
            plotted_field.lower_bound_in_GeV
            + (step_index * plotted_field.step_in_GeV)
        )

    potential_values = (
        QFT_model_configuration.potential_in_quartic_GeV_per_field_step[0]
    )

    CSV_writer.write_file_from_matrix(
    output_CSV_filename=data_filename,
        header_row=[field_name, "V"],
        value_matrix=[
            [field_in_GeV(i), potential_values[i]]
            for i in range(plotted_field.number_of_values)
        ]
    )

    with open(plotting_filename, "w") as output_file:
        output_file.write(
                    f"set title \"Potential of single field {field_name}\"\n"
                    "set datafile separator \";\"\n"
                    "set key autotitle columnhead\n"
                    "unset key\n"
                    "set xlabel \"{field_name} in GeV\"\n"
                    "set ylabel \"V({field_name}) in GeV^4\"\n"
                    "set term png\n"
                    f"set output \"{picture_filename}\"\n"
                    f"plot \"{data_filename}\" with linespoints"
                )
    import subprocess
    subprocess.call(
                f"{command_for_gnuplot} {plotting_filename}",
                shell=True
            )


if __name__ == "__main__":
    main()
