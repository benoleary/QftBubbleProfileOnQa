from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from dimod import SampleSet

from basis.field import FieldAtPoint, FieldCollectionAtPoint
import basis.variable
from dynamics.hamiltonian import AnnealerHamiltonian
from input.configuration import SpatialLatticeConfiguration
from minimization.sampling import SampleProvider
from minimization.weight import WeightAccumulator
from structure.domain_wall import DomainWallWeighter


# We would like to use kw_only=True, but that needs Python 3.10 or later.
@dataclass(frozen=True, repr=False, eq=False)
class ProfilePoint:
    radius_in_inverse_GeV: float
    first_field_strength_in_GeV: float
    second_field_strength_in_GeV: Optional[float]


class BubbleProfile:
    """
    This class represents the field or fields at all the points of the bubble
    profile.
    """
    def __init__(
            self,
            *,
            annealer_Hamiltonian: AnnealerHamiltonian,
            domain_wall_weighter: DomainWallWeighter,
            spatial_lattice_configuration: SpatialLatticeConfiguration
    ):
        """
        The constructor just sets up fields.
        """
        self.annealer_Hamiltonian = annealer_Hamiltonian
        self.domain_wall_weighter = domain_wall_weighter
        self.spatial_lattice_configuration = spatial_lattice_configuration
        spatial_name_function = basis.variable.name_for_index(
            "r",
            spatial_lattice_configuration.number_of_spatial_steps
        )

        self.first_field = annealer_Hamiltonian.get_first_field_definition()
        self.second_field = annealer_Hamiltonian.get_second_field_definition()

        def create_at_point(spatial_index: int) -> FieldCollectionAtPoint:
            return FieldCollectionAtPoint(
                spatial_point_identifier=spatial_name_function(spatial_index),
                spatial_radius_in_inverse_GeV=(
                    spatial_index
                    * spatial_lattice_configuration.spatial_step_in_inverse_GeV
                ),
                first_field=self.first_field,
                second_field=self.second_field
            )

        self.number_of_point_profiles = (
            spatial_lattice_configuration.number_of_spatial_steps + 1
        )
        self.fields_at_points = [
            create_at_point(spatial_index=i)
            for i in range(self.number_of_point_profiles)
        ]
        self.maximum_variable_weight = self._get_maximum_variable_weight()
        domain_wall_alignment_weight = 2.0 * self.maximum_variable_weight
        domain_end_fixing_weight = 2.0 * domain_wall_alignment_weight
        self.annealing_weights = self._set_up_weights(
            alignment_weight=domain_wall_alignment_weight,
            end_weight=domain_end_fixing_weight
        )

    def field_strengths_at_radius_values(
            self,
            *,
            solution_sample: SampleSet,
            sample_provider: SampleProvider
    ) -> Sequence[ProfilePoint]:
        spatial_step = (
            self.spatial_lattice_configuration.spatial_step_in_inverse_GeV
        )
        return [
           self._profile_at_radius_index(
               radius_index=i,
               spatial_step_in_inverse_GeV=spatial_step,
               fields_at_point=f,
               solution_sample=solution_sample,
               sample_provider=sample_provider
            )
           for i, f in enumerate(self.fields_at_points)
        ]

    def _profile_at_radius_index(
            self,
            *,
            radius_index: int,
            spatial_step_in_inverse_GeV: float,
            fields_at_point: FieldCollectionAtPoint,
            solution_sample: SampleSet,
            sample_provider: SampleProvider
    ) -> ProfilePoint:
        return ProfilePoint(
            radius_in_inverse_GeV=(radius_index * spatial_step_in_inverse_GeV),
            first_field_strength_in_GeV=(
                self._get_field_strength_in_GeV(
                    field_at_point=fields_at_point.first_field,
                    solution_sample=solution_sample,
                    sample_provider=sample_provider
                )
            ),
            second_field_strength_in_GeV=(
                None if not fields_at_point.second_field
                else self._get_field_strength_in_GeV(
                    field_at_point=fields_at_point.second_field,
                    solution_sample=solution_sample,
                    sample_provider=sample_provider
                )
            )
        )

    def _get_field_strength_in_GeV(
            self,
            *,
            field_at_point: FieldAtPoint,
            solution_sample: SampleSet,
            sample_provider: SampleProvider
    ) -> float:
        field_value_in_ones = (
            sample_provider.get_number_of_variables_in_one_state(
                variable_names=field_at_point.binary_variable_names,
                sample_set=solution_sample
            )
        )
        return (
            field_at_point.field_definition.lower_bound_in_GeV
            + (
                # We have to remove the first variable which is fixed to |1>.
                (field_value_in_ones - 1)
                * field_at_point.field_definition.step_in_GeV
            )
        )

    def _get_volume_factor(self, radius_value: float) -> float:
        return radius_value**(
            self.spatial_lattice_configuration.volume_exponent
        )

    def _get_maximum_variable_weight(self) -> float:
        maximum_kinetic = (
            self.annealer_Hamiltonian.get_maximum_kinetic_contribution(
                self.spatial_lattice_configuration.spatial_step_in_inverse_GeV
            )
        )
        # The largest volume factor in our calculation is from the derivative
        # between the edge and its neighbor, so the volume factor at the edge is
        # an upper bound on the volume factor and that is sufficient.
        maximum_volume_factor = self._get_volume_factor(
            self.fields_at_points[-1].spatial_radius_in_inverse_GeV
        )
        return (
            maximum_kinetic
            + self.annealer_Hamiltonian.get_maximum_potential_difference()
        ) * maximum_volume_factor

    def _set_up_weights(
            self,
            *,
            end_weight: float,
            alignment_weight: float
    ) -> WeightAccumulator:
        # We do the different terms in separate methods so that it is easier to
        # read.
        calculated_weights = self._get_structure_weights(
            end_weight=end_weight,
            alignment_weight=alignment_weight
        )
        calculated_weights.add(self._get_potential_weights())
        calculated_weights.add(
            self._get_kinetic_weights(
                self.spatial_lattice_configuration.spatial_step_in_inverse_GeV
            )
        )
        return calculated_weights

    def _get_structure_weights(
            self,
            *,
            end_weight: float,
            alignment_weight: float
    ) -> WeightAccumulator:
        calculated_weights = self._get_fixed_center_and_edge_weights(end_weight)
        calculated_weights.add(
            self.domain_wall_weighter.weights_for_domain_walls(
                profiles_at_points=self.fields_at_points[1:-1],
                end_weight=end_weight,
                alignment_weight=alignment_weight
            )
        )
        return calculated_weights

    def _get_fixed_center_and_edge_weights(
            self,
            end_weight:float
    ) -> WeightAccumulator:
        """
        This only works in the assumption of a single field which is set to
        1000... (or as many 1s as implied by the true vacuum in steps according
        to the field definition object) at the center and ...1110 (or as many 0s
        as implied by the false vacuum in steps according to the field
        definition object) at the edge, which is why the second field is
        ignored.
        """
        center_first_field = self.fields_at_points[0].first_field
        first_field_definition = center_first_field.field_definition
        calculated_weights = (
            self.domain_wall_weighter.weights_for_fixed_value(
                 field_at_point=center_first_field,
                 fixing_weight=end_weight,
                 number_of_ones=(
                    1 + first_field_definition.true_vacuum_value_in_steps
                )
            )
        )
        calculated_weights.add(
            self.domain_wall_weighter.weights_for_fixed_value(
                 field_at_point=self.fields_at_points[-1].first_field,
                 fixing_weight=end_weight,
                 number_of_ones=(
                    1 + first_field_definition.false_vacuum_value_in_steps
                )
            )
        )
        # TODO: enhance for second field
        return calculated_weights

    def _get_potential_weights(self) -> WeightAccumulator:
        calculated_biases = WeightAccumulator()

        # The fields at the bubble center and edge are fixed so it is not worth
        # evaluating the potential there.
        for profile_at_point in self.fields_at_points[1:-1]:
            volume_factor = self._get_volume_factor(
                profile_at_point.spatial_radius_in_inverse_GeV
            )
            calculated_biases.add(
                self.annealer_Hamiltonian.potential_weights(
                    first_field=profile_at_point.first_field,
                    second_field=profile_at_point.second_field,
                    scaling_factor=volume_factor
                )
            )

        return calculated_biases

    def _get_kinetic_weights(
            self,
            spatial_step: float
    ) -> WeightAccumulator:
        calculated_biases = WeightAccumulator()
        previous_profile = self.fields_at_points[0]
        for profile_at_point in self.fields_at_points[1:]:
            # There are many ways to discretize a term which depends on the
            # radius and a derivative with respect to the radius. We take the
            # derivative at the half-way points of each interval, and use the
            # radius value for that half-way point.
            average_radius = 0.5 * (
                profile_at_point.spatial_radius_in_inverse_GeV
                + previous_profile.spatial_radius_in_inverse_GeV
            )
            volume_factor = self._get_volume_factor(average_radius)
            calculated_biases.add(
                self.annealer_Hamiltonian.kinetic_weights(
                    radius_step_in_inverse_GeV=spatial_step,
                    nearer_center=previous_profile,
                    nearer_edge=profile_at_point,
                    scaling_factor=volume_factor
                )
            )
            previous_profile = profile_at_point
        return calculated_biases
