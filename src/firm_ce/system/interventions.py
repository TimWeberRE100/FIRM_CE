from typing import Dict


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
    """

    def __init__(
        self,
        intervention_id: int,
        name: str,
        generator_capacities: Dict[int, float],
        storage_capacities_p: Dict[int, float],
        storage_capacities_e: Dict[int, float],
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
        self.id = intervention_id
        self.name = name
        self.generator_capacities = generator_capacities
        self.storage_capacities_p = storage_capacities_p
        self.storage_capacities_e = storage_capacities_e

    def __repr__(self) -> str:
        return f"Intervention({self.id!r} {self.name!r})"


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

    def __init__(self, interventions: Dict[int, Dict[int, "Intervention"]]) -> None:
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
