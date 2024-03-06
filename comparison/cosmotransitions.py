import argparse

from cosmoTransitions.pathDeformation import fullTunneling as CT_full_tunneling
from cosmoTransitions.tunneling1D import SingleFieldInstanton

from potential import CtPotential


def for_single_field(ct_potential: CtPotential):
    bubble_profile = SingleFieldInstanton(
        phi_absMin=-1.0,
        phi_metaMin=1.0,
        V=ct_potential.get_single_field_potential(),
        dV=ct_potential.get_single_field_gradient(),
        alpha=3
    ).findProfile()

    content_for_CSV = (
        ["r in 1/GeV ; f in GeV"]
        + [
            f"{r} ; {f}"
            for r, f in zip(bubble_profile.R, bubble_profile.Phi)
        ]
    )
    data_filename = "temporary_gnuplot_input.csv"
    print(f"writing profile in {data_filename}")
    with open(data_filename, "w") as output_file:
        output_file.write("\n".join(content_for_CSV) + "\n")

    picture_filename = data_filename.rsplit(".", 1)[0] + ".png"
    plotting_filename = "temporary_gnuplot_input.in"
    print(
        f"running gnuplot on {plotting_filename}"
    )
    with open(plotting_filename, "w") as output_file:
        output_file.write(
            f"set title \"CosmoTransitions, single field\"\n"
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
        f"gnuplot {plotting_filename}",
        shell=True
    )


def for_two_fields(ct_potential: CtPotential):
    # We only check against an example where the true vacuum and false vacuum
    # are at the boundaries of the first field and at 0 for the second field in
    # both cases.
    true_vacuum = [
        -ct_potential.model_parameters.first_field_bound_in_GeV,
        0.0
    ]
    false_vacuum = [
        ct_potential.model_parameters.first_field_bound_in_GeV,
        0.0
    ]

    tunneling_result = CT_full_tunneling(
        path_pts=[true_vacuum, false_vacuum],
        V=ct_potential.get_two_field_potential(),
        dV=ct_potential.get_two_field_gradient(),
        tunneling_init_params={
            "alpha": 3 # 3 for quantum tunneling, 2 for thermal tunneling
        },
        tunneling_findProfile_params = {"npoints": 20},
        deformation_deform_params = {"maxiter": 10},
        maxiter = 10
    )

    content_for_CSV = (
        ["r in 1/GeV ; f in GeV ; g in GeV"]
        + [
            f"{r} ; {p[0]} ; {p[1]}"
            for r, p in zip(tunneling_result.profile1D.R, tunneling_result.Phi)
        ]
    )
    data_filename = "temporary_gnuplot_input.csv"
    print(f"writing profile in {data_filename}")
    with open(data_filename, "w") as output_file:
        output_file.write("\n".join(content_for_CSV) + "\n")

    picture_filename = data_filename.rsplit(".", 1)[0] + ".png"
    plotting_filename = "temporary_gnuplot_input.in"
    print(
        f"running gnuplot on {plotting_filename}"
    )
    with open(plotting_filename, "w") as output_file:
        output_file.write(
            f"set title \"CosmoTransitions, two fields\"\n"
            "set datafile separator \";\"\n"
            "set key autotitle columnhead\n"
            "unset key\n"
            "set xlabel \"r in 1/GeV\"\n"
            f"set ylabel \"g in GeV\"\n"
            f"set zlabel \"f in GeV\"\n"
            "set term png\n"
            f"set output \"{picture_filename}\"\n"
            f"splot \"{data_filename}\" using 1:3:2 with lines"
        )
    import subprocess
    subprocess.call(
        f"gnuplot {plotting_filename}",
        shell=True
    )


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_name")
    argument_parser.add_argument("--second_field", action="store_true")
    parsed_arguments = argument_parser.parse_args()

    ct_potential = CtPotential(
        model_name=parsed_arguments.model_name,
        has_second_field=parsed_arguments.second_field
    )

    if not parsed_arguments.second_field:
        for_single_field(ct_potential)
    else:
        for_two_fields(ct_potential)
