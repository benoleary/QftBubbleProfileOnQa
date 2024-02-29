# QftBubbleProfileOnQa

This is a Python program to minimize an integral of a Hamiltonian with boundary
conditions of being at the true vacuum at the lattice point denoting the origin
and being at the false vacuum at the other end of the lattice, using a quantum
annealer.

One can use models with either a single scalar field, or with two scalar fields.

According to https://arxiv.org/abs/2003.07374, this should yield the bubble
profile for the bounce action, which gives the decay probability per space-time
volume for a region of false vacuum in a quantum field theory, but as I was
implementing the design, I realized that this is completely wrong. The integral
over the Hamiltonian has a negative eigenvalue, and one has to find the
stationary point of the action by means other than minimization. There is no
way a minimization of the integral of the Hamiltonian would yield an unstable
state, which is what the critical bubble is, and the analytic continuation
calculation used by Callan, Curtis, and Coleman obtains the imaginary part of
the energy, which is exactly what one should get for an unstable state.

One can easily see that the integral of the Hamiltonian is unbounded from below,
as simple increasing the bubble radius will lower the energy by the difference
in volume multiplied by the difference in energy density between the true vacuum
and the false vacuum. The result published in https://arxiv.org/abs/2003.07374
seems to be completely due to the boundary conditions forcing the ends of the
lattice to the vacua, and it does not even agree with the using their numbers
with the known analytic results of Coleman - their bubble is 10-15 units in
radius, but it should be 1/epsilon, and their epsilon is 0.01, so the radius
should be 100 units.

However, https://arxiv.org/abs/2003.07374 does describe a correct program for a
quantum annealer to minimize the integral of the Hamiltonian, so feel free to
use this program to explore that mathematics problem which bears a superficial
resemblance to a different mathematics problem which can be used to constrain
the scalar field sector of quantum field theory models.


# Licence

This is open-source software under the Apache License Version 2.0 (see the file
LICENSE in the same root directory as this file), written by Ben O'Leary
(benjamin.oleary@gmail.com)

# Usage

In order to run this, you need to have the Ocean SDK from D-Wave installed.
You also need to have a D-Wave Leap account for submitting jobs to their
quantum annealers if you want to try it out on an actual quantum computer,
rather than using classical annealing or the other local options.

The program can be run as

python3 main.py /path/to/your/configuration.xml

with the configuration file being an XML file with the required information.
Please examine the examples in ./examples/ for how to prepare the file.
Importantly, the element potential_in_quartic_GeV_per_field_step should be the
numerical values of the potential per step in the field strength.

If using a single scalar field, the element content should be a
semicolon-separated list of values starting at the lowest value of the field
represented in the program up to the highest. For example, for a single field
which is resolved into 5 values in the program,
<potential_in_quartic_GeV_per_field_step>
    0.1;2.2;3.2;2.7;1.3
<potential_in_quartic_GeV_per_field_step>
is the correct form.

If using two scalar fields, the element content should be a
hash-character-separated list of semicolon-separated lists of values. Each
semicolon-separated list should be the form of the single-field case for the
first field as described above, for a fixed value of the second field. These
fixed values are the resolved values of the second field. For example, for
resolving the first field into 4 values and the second into 3 values,
<potential_in_quartic_GeV_per_field_step>
    1.1;2.2;3.3;4.4#5.5;6.6;7.7;8.8#9.9;10.0;11.0;12.0
<potential_in_quartic_GeV_per_field_step>
is the correct form.

There is a helper file for creating examples based on a potential energy
function defined in Python. Edit example.py accordingly, and run

python3 example.py --model_name sm --second_field

to use the SM-inspired parameters for two fields.

There is also a helper file for calculating the correct bubble profile using
CosmoTransitions by Carroll Wainwright, which can be run with

PYTHONPATH=/path/to/cosmotransitions/ python3 cosmotransitions.py --model_name acs --second_field

for parameters used in https://arxiv.org/abs/2003.07374 extended to using a
second field (but not the U(1) string case described in that paper).
