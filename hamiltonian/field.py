class Field:
    """
    This class represents the strength of a QFT scalar field at a point in
    space-time.
    """
    def __init__(
            self,
            field_name: str,
            spatial_point_identifier: str,
            number_of_binary_variables: input
        ):
        """
        The constructor just sets up the names for the binary variables, since
        the D-Wave samplers just want dicts of names of binary variables or
        pairs of names of binary variables, mapped to weights.
        """
        self.field_name = field_name
        self.spatial_point_identifier = spatial_point_identifier
        self.number_of_binary_variables = number_of_binary_variables
        self.binary_variable_names = [
            f"{field_name}_{spatial_point_identifier}_{binary_variable_index}"
            for binary_variable_index in range(number_of_binary_variables)
        ]
