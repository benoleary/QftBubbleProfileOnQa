from typing import Callable
from minimization.weight import AccumulatingWeightMatrix
from hamiltonian.field import FieldAtPoint

class SingleFieldPotential:
    """
    This class adds calculates weights for binary variables of a single field
    represented by a set of IcdwFields, one for every spatial point. (The
    calculation is significantly different when dealing with multiple fields at
    a single spatial point.)

    The field strength is represented by the lowest value (only the first binary
    variable is 1) being the value of the field at the center of the critical
    bubble, and by its highest value (only the last variable is 0) at the false
    vacuum. Another class will take care of translating this back to the units
    of the QFT problem.
    """
    def __init__(self, normalized_potential: Callable[[float], float]):
        self.normalized_potential = normalized_potential

    def get_weights_for_potential(
        self,
        single_field: FieldAtPoint
    ) -> AccumulatingWeightMatrix:
        weight_matrix = {}
        names_with_field_values = \
            single_field.binary_variable_names_with_field_values
        _, previous_field_value = names_with_field_values[0]
        previous_potential_value = \
            self.normalized_potential(previous_field_value)
        for variable_name, field_value in names_with_field_values[1:]:
            potential_value = self.normalized_potential(field_value)
            weight_matrix[(variable_name, variable_name)] = \
                potential_value - previous_potential_value
            previous_potential_value = potential_value
        return AccumulatingWeightMatrix(weight_matrix)
