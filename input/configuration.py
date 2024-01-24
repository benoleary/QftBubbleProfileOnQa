from typing import List, Optional
from dataclasses import dataclass
import xml.etree.ElementTree

from basis.field import FieldDefinition


def xml_str(
        *,
        parent_element: xml.etree.ElementTree,
        element_name: str
) -> Optional[str]:
        xml_element = parent_element.find(element_name)
        # The elements are not truthy in an intuitive way! We have to check
        # against None.
        if xml_element is None:
            return None
        return xml_element.text


def xml_int(
        *,
        parent_element: xml.etree.ElementTree,
        element_name: str
) -> Optional[int]:
    element_text = xml_str(
        parent_element=parent_element,
        element_name=element_name
    )
    return int(element_text) if element_text else None


def xml_float(
        *,
        parent_element: xml.etree.ElementTree,
        element_name: str
) -> Optional[float]:
    element_text = xml_str(
        parent_element=parent_element,
        element_name=element_name
    )
    return float(element_text) if element_text else None


class QftModelConfiguration:
    def __init__(
            self,
            *,
            first_field: FieldDefinition,
            second_field: Optional[FieldDefinition] = None,
            potential_in_quartic_GeV_per_field_step: List[List[float]]
    ):
        self.number_of_values_for_second_field = len(
            potential_in_quartic_GeV_per_field_step
        )
        self.number_of_values_for_first_field = (
            0 if not self.number_of_values_for_second_field
            else len(potential_in_quartic_GeV_per_field_step[0])
        )
        if not self.number_of_values_for_first_field:
            raise ValueError("Cannot have a potential without any values")
        if second_field and self.number_of_values_for_second_field < 2:
            raise ValueError(
                "Second field defined but only one row of potential values"
                " (i.e. there is only one value allowed for the second field)"
            )
        self.first_field = first_field
        self.second_field = second_field
        self.potential_in_quartic_GeV_per_field_step = (
            potential_in_quartic_GeV_per_field_step
        )


@dataclass(kw_only=True, frozen=True, repr=False, eq=False)
class SpatialLatticeConfiguration:
    number_of_spatial_steps: int
    spatial_step_in_inverse_GeV: float
    volume_exponent: int


@dataclass(kw_only=True, frozen=True, repr=False, eq=False)
class AnnealerConfiguration:
    variable_type: str = "bit"
    sampler_name: str = "default"
    number_of_shots: Optional[int] = None


@dataclass(kw_only=True, frozen=True, repr=False, eq=False)
class OutputConfiguration:
    output_CSV_filename: Optional[str] = None
    command_for_gnuplot: Optional[str] = None


class FullConfiguration:
    def __init__(
            self,
            input_filename: str
    ):
        input_xml_root = xml.etree.ElementTree.parse(input_filename).getroot()

        qft_element = input_xml_root.find("qft")

        def qft_xml_field_definition(
                *,
                element_name: str,
                number_of_values: int
        ) -> Optional[float]:
            field_element = qft_element.find(element_name)
            # The elements are not truthy in an intuitive way! We have to check
            # against None.
            if field_element is None:
                return None
            return FieldDefinition(
                    field_name=xml_str(
                        parent_element=field_element,
                        element_name="field_name"
                    ),
                    number_of_values=number_of_values,
                    lower_bound_in_GeV=xml_float(
                        parent_element=field_element,
                        element_name="lower_bound_in_GeV"
                    ),
                    upper_bound_in_GeV=xml_float(
                        parent_element=field_element,
                        element_name="upper_bound_in_GeV"
                    ),
                    true_vacuum_value_in_GeV=xml_float(
                        parent_element=field_element,
                        element_name="true_vacuum_value_in_GeV"
                    ),
                    false_vacuum_value_in_GeV=xml_float(
                        parent_element=field_element,
                        element_name="false_vacuum_value_in_GeV"
                    )
                )

        first_field = qft_xml_field_definition(
            element_name="first_field",
            number_of_values=len(potential_per_field_step[0])
        )
        if first_field is None:
            raise ValueError("No XML element for first field")

        second_field = qft_xml_field_definition(
            element_name="second_field",
            number_of_values=len(potential_per_field_step)
        )

        potential_element = qft_element.find(
            "potential_in_quartic_GeV_per_field_step"
        )

        if potential_element is None:
            raise ValueError("No XML element for potential")
        potential_per_field_step=[
            [float(v) for v in potential_row.split(";")]
            for potential_row in potential_element.text.split("#")
        ]

        self.QFT_model_configuration = QftModelConfiguration(
            first_field=first_field,
            second_field=second_field,
            potential_in_quartic_GeV_per_field_step=potential_element
        )

        space_element = input_xml_root.find("space")
        self.spatial_lattice_configuration = SpatialLatticeConfiguration(
            number_of_spatial_steps=xml_int(
                parent_element=space_element,
                element_name="number_of_spatial_steps"
            ),
            spatial_step_in_inverse_GeV=xml_float(
                parent_element=space_element,
                element_name="spatial_step_in_inverse_GeV"
            ),
            volume_exponent=xml_int(
                parent_element=space_element,
                element_name="volume_exponent"
            )
        )

        annealer_element = input_xml_root.find("annealer")
        self.annealer_configuration = AnnealerConfiguration(
            variable_type=xml_str(
                parent_element=annealer_element,
                element_name="variable_type"
            ),
            sampler_name=xml_str(
                parent_element=annealer_element,
                element_name="sampler_name"
            ),
            number_of_shots=xml_int(
                parent_element=annealer_element,
                element_name="number_of_shots"
            )
        )

        output_element = input_xml_root.find("output")
        self.output_configuration = OutputConfiguration(
            output_CSV_filename=xml_str(
                parent_element=output_element,
                element_name="output_CSV_filename"
            ),
            command_for_gnuplot=xml_str(
                parent_element=output_element,
                element_name="command_for_gnuplot"
            )
        )
