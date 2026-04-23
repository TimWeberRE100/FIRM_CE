from typing import Dict, Tuple

from firm_ce.common.constants import NP_INT64_MAX
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.fast_methods.ltcosts_m import calculate_annualised_build_func

class Intervention:
    """
    Represents one named intervention group from interventions.csv for a scenario.

    An Intervention bundles a set of generators and storages together with their
    discrete build step sizes. Asset membership is expressed as capacity dictionaries
    keyed by scenario-level order (the same key used in Fleet.generators /
    Fleet.storages).

    Attributes:
    -------
    id (int): Model-level identifier matching the id column in interventions.csv.
    name (str): Human-readable name from interventions.csv.
    generator_capacities (Dict[int, float]): Discrete power build step (GW) for each
        generator in this intervention, keyed by generator.order.
    storage_capacities_p (Dict[int, float]): Discrete power build step (GW) for each
        storage in this intervention, keyed by storage.order.
    storage_capacities_e (Dict[int, float]): Discrete energy build step (GWh) for each
        storage in this intervention, keyed by storage.order. Empty dict when no storage
        energy interventions are defined (e.g. all storages have fixed duration).
    maximum_repetitions (int): Maximum allowable repetitions of the intervention in a 
        single intervention set. Constrained by maximum build limits of the assets.
        Initialised to 0 and dynamically updated when solving the pathway_planning problem.
    """

    def __init__(
        self,
        intervention_id: int,
        name: str,
        generator_capacities: Dict[int, float],
        storage_capacities_p: Dict[int, float],
        storage_capacities_e: Dict[int, float],
        generator_max_build_step: Dict[int, float],
        storage_max_build_p_step: Dict[int, float],
        storage_max_build_e_step: Dict[int, float],
        fleet: Fleet_InstanceType,
        year_idx: int
    ) -> None:
        """
        Initialise an Intervention instance.

        Parameters:
        -------
        intervention_id (int): Model-level identifier.
        name (str): Human-readable name.
        generator_capacities (Dict[int, float]): Power build steps keyed by generator.order.
        storage_capacities_p (Dict[int, float]): Power build steps keyed by storage.order.
        storage_capacities_e (Dict[int, float]): Energy build steps keyed by storage.order.

        Returns:
        -------
        None.
        """
        self.idx = intervention_id
        self.year_idx = year_idx
        self.name = name
        self.capacities = (generator_capacities, storage_capacities_p, storage_capacities_e)
        self.max_build_step = (generator_max_build_step, storage_max_build_p_step, storage_max_build_e_step)
        self.annualised_costs = self.set_annualised_costs(fleet)
        self.maximum_repetitions = self.set_maximum_repetitions()

    def __repr__(self) -> str:
        return f"Intervention({self.idx!r} {self.name!r})"
    
    def set_annualised_costs(self, fleet: Fleet_InstanceType) -> Tuple[Dict[int,float], Dict[int,float]]:
        generator_capacities, storage_capacities_p, storage_capacities_e = self.capacities
        generator_annualised_costs = {}
        storage_p_annualised_costs = {}
        storage_e_annualised_costs = {}
        for order, capacity in generator_capacities.items():
            generator_annualised_costs[order] = calculate_annualised_build_func(
                0,
                capacity,
                0,
                fleet.generators[order].cost[self.year_idx],
                1,
                "generator",
            )

        for order, capacity_p in storage_capacities_p.items():
            storage_p_annualised_costs[order] = calculate_annualised_build_func(
                0,
                capacity_p,
                0,
                fleet.storages[order].cost[self.year_idx],
                1,
                "storage",
            ) 

        for order, capacity_e in storage_capacities_e.items():
            storage_e_annualised_costs[order] = calculate_annualised_build_func(
                capacity_e,
                0,
                0,
                fleet.storages[order].cost[self.year_idx],
                1,
                "storage",
            )            

        return generator_annualised_costs, storage_p_annualised_costs, storage_e_annualised_costs

    def set_maximum_repetitions(self) -> int:
        maximum_repetitions = NP_INT64_MAX
        generator_capacities, storage_capacities_p, storage_capacities_e = self.capacities

        for order, capacity in generator_capacities.items():
            if capacity > 0:
                maximum_repetitions = min(int(self.max_build_step[0][order] // capacity), maximum_repetitions)
                if not maximum_repetitions > 0:
                    break

        for order, capacity_p in storage_capacities_p.items():
            if capacity_p > 0:
                maximum_repetitions = min(int(self.max_build_step[1][order] // capacity_p), maximum_repetitions)
                if not maximum_repetitions > 0:
                    break

        for order, capacity_e in storage_capacities_e.items():
            if capacity_e > 0:
                maximum_repetitions = min(int(self.max_build_step[2][order] // capacity_e), maximum_repetitions)
                if not maximum_repetitions > 0:
                    break
        return maximum_repetitions

class ScenarioInterventions:
    """
    Container for all Intervention objects applicable to a scenario, grouped by investment year.

    Attributes:
    -------
    interventions (Dict[int, Dict[int, Intervention]]): For each investment year index
        (0-based offset from firstyear), a dict of Intervention objects keyed by
        model-level intervention id. Capacities in each Intervention are the cumulative
        sum of build steps from the previous investment year (exclusive) up to and
        including the current one.
    """

    def __init__(self, interventions: Dict[int, Dict[int, Intervention]]) -> None:
        """
        Initialise an Interventions container.

        Parameters:
        -------
        interventions (Dict[int, Dict[int, Intervention]]): Intervention objects indexed
            first by investment year index (offset from firstyear) and then by
            model-level intervention id.

        Returns:
        -------
        None.
        """
        self.interventions = interventions

    def __repr__(self) -> str:
        return f"ScenarioInterventions({list(self.interventions.keys())!r})"
