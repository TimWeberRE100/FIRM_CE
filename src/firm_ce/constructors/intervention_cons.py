from typing import Any, Dict

from numpy.typing import NDArray

from firm_ce.common.helpers import parse_id_list, parse_float_list
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.interventions import Intervention, ScenarioInterventions


def construct_ScenarioInterventions_object(
    interventions_imported_dict: Dict[int, Dict[int, Dict[str, Any]]],
    fleet: Fleet_InstanceType,
    firstyear: int,
    finalyear: int,
    investment_steps: NDArray,
) -> ScenarioInterventions:
    """
    Build an ScenarioInterventions container from the scenario-filtered, year-expanded interventions
    data and a constructed Fleet instance.

    For each investment step year, capacities are accumulated as the cumulative sum of all
    yearly build steps from the previous investment step (exclusive) up to and including the
    current step. The resulting ScenarioInterventions dict is keyed by investment year index
    (0-based offset from firstyear).

    Each intervention's generator_ids and storage_ids (model-level) are resolved to
    scenario-level order keys by checking the corresponding objects in the Fleet instance.

    Parameters:
    -------
    interventions_imported_dict (Dict[int, Dict[int, Dict[str, Any]]]): Scenario-filtered,
        year-expanded interventions data as returned by Scenario.get_scenario_dicts. Keyed by
        intervention id, then by year integer.
    fleet (Fleet_InstanceType): The constructed Fleet instance for this scenario, used to
        resolve model-level generator/storage ids to scenario-level order keys.
    firstyear (int): First year of the modelling horizon.
    finalyear (int): Final year of the modelling horizon.
    investment_steps (NDArray): Sorted int64 array of years in which investments are made.

    Returns:
    -------
    ScenarioInterventions: Container of Intervention objects indexed by investment year index and
        intervention id. Returns an ScenarioInterventions instance with an empty dict when
        interventions_imported_dict is empty or investment_steps is empty.
    """
    if not interventions_imported_dict or len(investment_steps) == 0:
        return ScenarioInterventions(interventions={})

    generator_id_to_order = {int(g.id): int(g.order) for g in fleet.generators.values()}
    storage_id_to_order = {int(s.id): int(s.order) for s in fleet.storages.values()}

    sorted_investment_steps = sorted(int(s) for s in investment_steps)
    interventions_for_investment_steps: Dict[int, Dict[int, Intervention]] = {}

    for step_pos, step_end_year in enumerate(sorted_investment_steps):
        investment_year_idx = step_end_year - firstyear
        step_start_year = sorted_investment_steps[step_pos - 1] + 1 if step_pos > 0 else firstyear

        step_interventions: Dict[int, Intervention] = {}

        for intervention_id, intervention_year_dict in interventions_imported_dict.items():
            any_year_data = next(iter(intervention_year_dict.values()))
            name = str(any_year_data.get("name", ""))

            cumulative_generator_capacities: Dict[int, float] = {}
            cumulative_storage_capacities_p: Dict[int, float] = {}
            cumulative_storage_capacities_e: Dict[int, float] = {}
            cumulative_generator_max_build: Dict[int, float] = {}
            cumulative_storage_max_build_p: Dict[int, float] = {}
            cumulative_storage_max_build_e: Dict[int, float] = {}

            for year in range(step_start_year, step_end_year + 1):
                year_idx = year - firstyear
                intervention_year_data = intervention_year_dict[year]
                generator_ids = parse_id_list(intervention_year_data["generator_ids"])
                storage_ids = parse_id_list(intervention_year_data["storage_ids"])
                generator_capacities = parse_float_list(intervention_year_data["generator_intervention_capacities"])
                storage_capacities_p = parse_float_list(intervention_year_data["storage_intervention_capacities_p"])
                storage_capacities_e = parse_float_list(intervention_year_data["storage_intervention_capacities_e"])

                for generator_idx, capacity in zip(generator_ids, generator_capacities):
                    order = generator_id_to_order[generator_idx]
                    cumulative_generator_capacities[order] = cumulative_generator_capacities.get(order, 0.0) + capacity
                    cumulative_generator_max_build[order] = cumulative_generator_max_build.get(order, 0.0) + fleet.generators[order].max_build[year_idx]

                for storage_idx, capacity_p, capacity_e in zip(storage_ids, storage_capacities_p, storage_capacities_e):
                    order = storage_id_to_order[storage_idx]
                    cumulative_storage_capacities_p[order] = cumulative_storage_capacities_p.get(order, 0.0) + capacity_p
                    cumulative_storage_capacities_e[order] = cumulative_storage_capacities_e.get(order, 0.0) + capacity_e
                    cumulative_storage_max_build_p[order] = cumulative_storage_max_build_p.get(order, 0.0) + fleet.storages[order].max_build_p[year_idx]
                    cumulative_storage_max_build_e[order] = cumulative_storage_max_build_e.get(order, 0.0) + fleet.storages[order].max_build_e[year_idx]

            step_interventions[intervention_id] = Intervention(
                intervention_id=intervention_id,
                name=name,
                generator_capacities=cumulative_generator_capacities,
                storage_capacities_p=cumulative_storage_capacities_p,
                storage_capacities_e=cumulative_storage_capacities_e,
                generator_max_build_step=cumulative_generator_max_build,
                storage_max_build_p_step=cumulative_storage_max_build_p,
                storage_max_build_e_step=cumulative_storage_max_build_e,
                fleet=fleet,
                year_idx=investment_year_idx
            )

        interventions_for_investment_steps[investment_year_idx] = step_interventions

    return ScenarioInterventions(interventions=interventions_for_investment_steps)
