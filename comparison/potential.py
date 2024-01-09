import numpy
import parameters

parameters_for_ACS = parameters.for_ACS(
    N=50,
    M=50
)
(
    first_field_offset_in_GeV,
    first_field_step_in_GeV,
    spatial_step_in_inverse_GeV,
    field_to_GeV,
    potential_in_quartic_GeV_from_field_in_GeV,
    number_of_field_values,
    number_of_spatial_steps
) = parameters_for_ACS
half_step_size_in_GeV = 0.01
inverse_step_size = 0.5 / half_step_size_in_GeV


def call_on_unpacked(X, float_to_float):
    if ((1,) == X.shape ):
        return float_to_float(X)
    elif ((len(X), 1) == X.shape):
        returnArray = numpy.zeros(len(X))
        for i in range(len(X) ):
            returnArray[i] = float_to_float(X[i])
        return returnArray
    return None


def gradient_for_single_point(f):
    half_step_lower = potential_in_quartic_GeV_from_field_in_GeV(
        f - half_step_size_in_GeV
    )
    half_step_upper = potential_in_quartic_GeV_from_field_in_GeV(
        f + half_step_size_in_GeV
    )
    return (half_step_upper - half_step_lower) * inverse_step_size


def ct_potential(X):
    return potential_in_quartic_GeV_from_field_in_GeV(X)


def ct_gradient(X):
    return gradient_for_single_point(X)
