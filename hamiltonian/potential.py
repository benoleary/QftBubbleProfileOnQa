from minimization.weight import AccumulatingWeightMatrix
from hamiltonian.field import FieldAtPoint
from configuration.configuration import DiscreteConfiguration

def weights_for(
        configuration: DiscreteConfiguration,
        single_field: FieldAtPoint
    ) -> AccumulatingWeightMatrix:
        """
        This function adds calculates weights for binary variables of a single
        field represented by a set of IcdwFields, one for every spatial point.
        (The calculation is significantly different when dealing with multiple
        fields at a single spatial point.)
        """
        potential_values = configuration.potential_in_quartic_GeV_per_field_step
        number_of_potential_values = len(potential_values)
        # If there are N binary variables for the field, at least 1 and at most
        # (N - 1) of them will be 1 and the rest 0, so there are (N - 1) values
        # possible for the field, which must match the number of values provided
        # for the potential.
        number_of_field_values = len(single_field.binary_variable_names) - 1
        if number_of_potential_values != number_of_field_values:
            raise ValueError(
                f"Provided {number_of_potential_values} values for potential"
                f" but {number_of_field_values} values for field"
            )
        weight_matrix = {}
        # We never actually use the first value of the potential - its constant
        # value is implicitly subtracted from the Hamiltonian. So if we have 4
        # values U_0, U_1, U_2, and U_3, the field has 5 binary variables, where
        # the first and last are fixed, and bitstrings 10000 has implicit value
        # 0 = U_0 - U_0, 11000 has U_1 - U_0, 11100 has U_2 - U_1, and 11110 has
        # U_3 - U_2. (There are no other valid bitstrings for this field.)
        for i in range(1, number_of_field_values):
            variable_name = single_field.binary_variable_names[i]
            weight_matrix[(variable_name, variable_name)] = (
                potential_values[i] - potential_values[i - 1]
            )
        return AccumulatingWeightMatrix(weight_matrix)
