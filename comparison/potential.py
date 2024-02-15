import numpy

import parameters


class CtPotential:
    def __init__(
        self,
        *,
        model_name: str,
        has_second_field: bool
    ):
        self.model_parameters = parameters.for_model(
            model_name=model_name,
            has_second_field=has_second_field
        )
        self.half_step_size_in_GeV = 0.01
        self.inverse_step_size = 0.5 / self.half_step_size_in_GeV
        self.potential_function = (
            self.model_parameters.potential_in_quartic_GeV_from_fields_in_GeV
        )

    def call_on_unpacked(self, X, for_point, has_array_elements):
        # This is only relevant for the case of two fields.
        if ((2,) == X.shape):
            return for_point(X)
        elif ((len(X), 2) == X.shape):
            # We have to work around numpy's requirements for efficiency given
            # how CosmoTransitions mixes up types.
            numpy_object = (
                X.copy() if has_array_elements else numpy.zeros(len(X))
            )
            for i in range(len(X)):
                numpy_object[i] = for_point(X[i])
            return numpy_object
        return None

    def get_single_field_potential(self):
        def V(x):
            return self.potential_function(x, 0.0)
        return V

    def get_single_field_gradient(self):
        def dV(x):
            half_step_lower = (
                self.potential_function(x - self.half_step_size_in_GeV, 0.0)
            )
            half_step_upper = (
                self.potential_function(x + self.half_step_size_in_GeV, 0.0)
            )
            return (half_step_upper - half_step_lower) * self.inverse_step_size
        return dV

    def get_two_field_potential(self):
        def underlying_V(X):
            return self.potential_function(X[0], X[1])
        def V(X):
            return self.call_on_unpacked(X, underlying_V, False)
        return V

    def get_two_field_gradient(self):
        def underlying_dV(X):
            first_half_step_lower = (
                self.potential_function(X[0] - self.half_step_size_in_GeV, X[1])
            )
            first_half_step_upper = (
                self.potential_function(X[0] + self.half_step_size_in_GeV, X[1])
            )
            first_difference = (first_half_step_upper - first_half_step_lower)
            second_half_step_lower = (
                self.potential_function(X[0], X[1] - self.half_step_size_in_GeV)
            )
            second_half_step_upper = (
                self.potential_function(X[0], X[1] + self.half_step_size_in_GeV)
            )
            second_difference = (
                second_half_step_upper - second_half_step_lower
                )
            return [
                first_difference * self.inverse_step_size,
                second_difference * self.inverse_step_size
            ]
        def dV(X):
            return self.call_on_unpacked(X, underlying_dV, True)
        return dV
