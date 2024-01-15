from cosmoTransitions import tunneling1D as CTT1
import potential as P

bubble_profile = CTT1.SingleFieldInstanton(
    -1.0,
    1.0,
    P.ct_potential,
    P.ct_gradient
).findProfile()

# print(f"bubble_profile = {bubble_profile}")
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
        f"set title \"CosmoTransitions\"\n"
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
