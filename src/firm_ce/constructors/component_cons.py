from typing import Any, Dict

import numpy as np

from firm_ce.common.logging import get_logger
from firm_ce.common.typing import DictType, TypedDict, float64, int64
from firm_ce.constructors.cost_cons import construct_UnitCost_object
from firm_ce.system.components import (
    Fleet,
    Fleet_InstanceType,
    Fuel,
    Fuel_InstanceType,
    Generator,
    Generator_InstanceType,
    Storage,
    Storage_InstanceType,
)
from firm_ce.system.costs import UnitCost_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType


def construct_Fuel_object(
    fuel_id: int,
    fuel_year_dict: Dict[int, Dict[str, Any]],
    firstyear: int,
    finalyear: int,
) -> Fuel_InstanceType:
    """
    Takes a year-keyed fuel dictionary, builds year-keyed TypedDicts for cost and emissions,
    and returns a static instance of the Fuel jitclass.

    Parameters:
    -------
    fuel_id (int): The model-level id for the fuel.
    fuel_year_dict (Dict[int, Dict[str, Any]]): A dictionary of per-year fuel attribute dicts,
        keyed by year integer.
    firstyear (int): First year of the modelling horizon (inclusive).
    finalyear (int): Final year of the modelling horizon (inclusive).

    Returns:
    -------
    Fuel_InstanceType: A static instance of the Fuel jitclass.
    """
    any_year_data = next(iter(fuel_year_dict.values()))
    name = str(any_year_data["name"])

    cost = TypedDict.empty(key_type=int64, value_type=float64)
    emissions = TypedDict.empty(key_type=int64, value_type=float64)

    for year_idx, year in enumerate(range(firstyear, finalyear + 1)):
        if year in fuel_year_dict:
            cost[year_idx] = float(fuel_year_dict[year]["cost"])
            emissions[year_idx] = float(fuel_year_dict[year]["emissions"])
        else:
            get_logger().error(f"Year {year} data missing for Fuel id {fuel_id}.")
            raise ValueError(f"Year {year} data missing for Fuel id {fuel_id}.")

    return Fuel(True, fuel_id, name, cost, emissions)


def construct_Generator_object(
    generator_id: int,
    generator_year_dict: Dict[int, Dict[str, Any]],
    fuel_instances_by_name: Dict[str, Fuel_InstanceType],
    fuels_raw_by_name: Dict[str, Dict[int, Dict[str, Any]]],
    nodes_object_dict: DictType(int64, Node_InstanceType),
    lines_object_dict: DictType(int64, Line_InstanceType),
    order: int,
    firstyear: int,
    finalyear: int,
) -> Generator_InstanceType:
    """
    Takes data required to initialise a single Generator object, builds year-keyed TypedDicts
    for all year-varying attributes, and returns a static instance of the Generator jitclass.

    Parameters:
    -------
    generator_id (int): The model-level id for the generator.
    generator_year_dict (Dict[int, Dict[str, Any]]): A dictionary of per-year generator attribute dicts,
        keyed by year integer.
    fuel_instances_by_name (Dict[str, Fuel_InstanceType]): Mapping of fuel name to pre-built
        Fuel jitclass instances.
    fuels_raw_by_name (Dict[str, Dict[int, Dict[str, Any]]]): Mapping of fuel name to raw
        year-keyed fuel attribute dicts, used to compute per-year UnitCost.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of all Node
        jitclass instances for the scenario, keyed by Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of all Line
        jitclass instances for the scenario, keyed by Line.order.
    order (int): The scenario-specific id for the Generator instance.
    firstyear (int): First year of the modelling horizon (inclusive).
    finalyear (int): Final year of the modelling horizon (inclusive).

    Returns:
    -------
    Generator_InstanceType: A static instance of the Generator jitclass.
    """
    any_year_data = next(iter(generator_year_dict.values()))
    name = str(any_year_data["name"])
    unit_type = str(any_year_data["unit_type"])
    near_optimum_check = str(any_year_data.get("near_optimum", "")).lower() in ("true", "1", "yes")

    raw_group = any_year_data.get("range_group", "")
    if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == "":
        group = name
    else:
        group = str(raw_group).strip()

    node_name = str(any_year_data["node"])
    node = next((n for n in nodes_object_dict.values() if n.name == node_name), None)
    if node is None:
        get_logger().error(f"Node '{node_name}' not found for Generator id {generator_id}.")
        raise ValueError(f"Node '{node_name}' not found for Generator id {generator_id}.")

    fuel_name = str(any_year_data["fuel"])
    fuel = fuel_instances_by_name.get(fuel_name)
    if fuel is None:
        get_logger().error(f"Fuel '{fuel_name}' not found for Generator id {generator_id}.")
        raise ValueError(f"Fuel '{fuel_name}' not found for Generator id {generator_id}.")

    line_name = str(any_year_data["line"])
    line = next((ln for ln in lines_object_dict.values() if ln.name == line_name), None)
    if line is None:
        get_logger().error(f"Line '{line_name}' not found for Generator id {generator_id}.")
        raise ValueError(f"Line '{line_name}' not found for Generator id {generator_id}.")

    unit_size = TypedDict.empty(key_type=int64, value_type=float64)
    max_build = TypedDict.empty(key_type=int64, value_type=float64)
    min_build = TypedDict.empty(key_type=int64, value_type=float64)
    initial_capacity = TypedDict.empty(key_type=int64, value_type=float64)
    cost = TypedDict.empty(key_type=int64, value_type=UnitCost_InstanceType)

    capacity = 0.0

    for year_idx, year in enumerate(range(firstyear, finalyear + 1)):
        if year in generator_year_dict:
            yr = generator_year_dict[year]
            unit_size[year_idx] = float(yr["unit_size"])
            max_build[year_idx] = float(yr["max_build"])
            min_build[year_idx] = float(yr["min_build"])
            initial_capacity[year_idx] = float(yr["initial_capacity"])

            raw_fuel_cost = 0.0
            if fuel_name in fuels_raw_by_name and year in fuels_raw_by_name[fuel_name]:
                raw_fuel_cost = float(fuels_raw_by_name[fuel_name][year]["cost"])

            cost[year_idx] = construct_UnitCost_object(
                capex_p=float(yr["capex"]),
                fom=float(yr["fom"]),
                vom=float(yr["vom"]),
                lifetime=int(yr["lifetime"]),
                discount_rate=float(yr["discount_rate"]),
                heat_rate_base=float(yr["heat_rate_base"]),
                heat_rate_incr=float(yr["heat_rate_incr"]),
                fuel_cost=raw_fuel_cost,
            )

            if year_idx == 0:
                capacity = float(yr["initial_capacity"])
        else:
            get_logger().error(f"Year {year} data missing for Generator id {generator_id}.")
            raise ValueError(f"Year {year} data missing for Generator id {generator_id}.")

    return Generator(
        True,
        generator_id,
        order,
        name,
        unit_size,
        max_build,
        min_build,
        initial_capacity,
        capacity,
        unit_type,
        near_optimum_check,
        node,
        fuel,
        line,
        group,
        cost,
    )


def construct_Storage_object(
    storage_id: int,
    storage_year_dict: Dict[int, Dict[str, Any]],
    nodes_object_dict: DictType(int64, Node_InstanceType),
    lines_object_dict: DictType(int64, Line_InstanceType),
    order: int,
    firstyear: int,
    finalyear: int,
) -> Storage_InstanceType:
    """
    Takes data required to initialise a single Storage object, builds year-keyed TypedDicts
    for all year-varying attributes, and returns a static instance of the Storage jitclass.

    Parameters:
    -------
    storage_id (int): The model-level id for the storage system.
    storage_year_dict (Dict[int, Dict[str, Any]]): A dictionary of per-year storage attribute dicts,
        keyed by year integer.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of all Node
        jitclass instances for the scenario, keyed by Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of all Line
        jitclass instances for the scenario, keyed by Line.order.
    order (int): The scenario-specific id for the Storage instance.
    firstyear (int): First year of the modelling horizon (inclusive).
    finalyear (int): Final year of the modelling horizon (inclusive).

    Returns:
    -------
    Storage_InstanceType: A static instance of the Storage jitclass.
    """
    any_year_data = next(iter(storage_year_dict.values()))
    name = str(any_year_data["name"])
    unit_type = str(any_year_data["unit_type"])
    near_optimum_check = str(any_year_data.get("near_optimum", "")).lower() in ("true", "1", "yes")

    raw_group = any_year_data.get("range_group", "")
    if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == "":
        group = name
    else:
        group = str(raw_group).strip()

    initial_power_capacity = TypedDict.empty(key_type=int64, value_type=float64)
    node_name = str(any_year_data["node"])
    node = next((n for n in nodes_object_dict.values() if n.name == node_name), None)
    if node is None:
        get_logger().error(f"Node '{node_name}' not found for Storage id {storage_id}.")
        raise ValueError(f"Node '{node_name}' not found for Storage id {storage_id}.")

    line_name = str(any_year_data["line"])
    line = next((ln for ln in lines_object_dict.values() if ln.name == line_name), None)
    if line is None:
        get_logger().error(f"Line '{line_name}' not found for Storage id {storage_id}.")
        raise ValueError(f"Line '{line_name}' not found for Storage id {storage_id}.")

    initial_energy_capacity = TypedDict.empty(key_type=int64, value_type=float64)
    duration = TypedDict.empty(key_type=int64, value_type=int64)
    charge_efficiency = TypedDict.empty(key_type=int64, value_type=float64)
    discharge_efficiency = TypedDict.empty(key_type=int64, value_type=float64)
    max_build_p = TypedDict.empty(key_type=int64, value_type=float64)
    max_build_e = TypedDict.empty(key_type=int64, value_type=float64)
    min_build_p = TypedDict.empty(key_type=int64, value_type=float64)
    min_build_e = TypedDict.empty(key_type=int64, value_type=float64)
    cost = TypedDict.empty(key_type=int64, value_type=UnitCost_InstanceType)

    power_capacity = 0.0
    energy_capacity = 0.0

    for year_idx, year in enumerate(range(firstyear, finalyear + 1)):
        if year in storage_year_dict:
            yr = storage_year_dict[year]
            power_cap = float(yr["initial_power_capacity"])
            dur = int(yr["duration"])
            if dur > 0:
                energy_cap = float(power_cap * dur)
            else:
                energy_cap = float(yr["initial_energy_capacity"])
                dur = int(energy_cap / power_cap) if power_cap > 0 else 0

            initial_power_capacity[year_idx] = power_cap
            initial_energy_capacity[year_idx] = energy_cap
            duration[year_idx] = dur
            charge_efficiency[year_idx] = float(yr["charge_efficiency"])
            discharge_efficiency[year_idx] = float(yr["discharge_efficiency"])
            max_build_p[year_idx] = float(yr["max_build_p"])
            max_build_e[year_idx] = float(yr["max_build_e"])
            min_build_p[year_idx] = float(yr["min_build_p"])
            min_build_e[year_idx] = float(yr["min_build_e"])

            cost[year_idx] = construct_UnitCost_object(
                capex_p=float(yr["capex_p"]),
                fom=float(yr["fom"]),
                vom=float(yr["vom"]),
                lifetime=int(yr["lifetime"]),
                discount_rate=float(yr["discount_rate"]),
                capex_e=float(yr["capex_e"]),
            )

            if year_idx == 0:
                power_capacity = power_cap
                energy_capacity = energy_cap
        else:
            get_logger().error(f"Year {year} data missing for Storage id {storage_id}.")
            raise ValueError(f"Year {year} data missing for Storage id {storage_id}.")

    return Storage(
        True,
        storage_id,
        order,
        name,
        initial_power_capacity,
        initial_energy_capacity,
        duration,
        charge_efficiency,
        discharge_efficiency,
        max_build_p,
        max_build_e,
        min_build_p,
        min_build_e,
        power_capacity,
        energy_capacity,
        unit_type,
        near_optimum_check,
        node,
        line,
        group,
        cost,
    )


def construct_Fleet_object(
    generators_imported_dict: Dict[int, Dict[int, Dict[str, Any]]],
    storages_imported_dict: Dict[int, Dict[int, Dict[str, Any]]],
    fuels_imported_dict: Dict[int, Dict[int, Dict[str, Any]]],
    lines_object_dict: DictType(int64, Line_InstanceType),
    nodes_object_dict: DictType(int64, Node_InstanceType),
    firstyear: int,
    finalyear: int,
) -> Fleet_InstanceType:
    """
    Takes data required to initialise a single Fleet object, casts values into Numba-compatible
    types, and returns an instance of the Fleet jitclass. The Fleet consists of all assets that
    can be dispatched in the energy balance.

    Parameters:
    -------
    generators_imported_dict (Dict[int, Dict[int, Dict[str, Any]]]): A year-keyed dictionary of
        generator attribute dicts, grouped by generator id.
    storages_imported_dict (Dict[int, Dict[int, Dict[str, Any]]]): A year-keyed dictionary of
        storage attribute dicts, grouped by storage id.
    fuels_imported_dict (Dict[int, Dict[int, Dict[str, Any]]]): A year-keyed dictionary of fuel
        attribute dicts, grouped by fuel id.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of all Line
        jitclass instances for the scenario, keyed by Line.order.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of all Node
        jitclass instances for the scenario, keyed by Node.order.
    firstyear (int): First year of the modelling horizon (inclusive).
    finalyear (int): Final year of the modelling horizon (inclusive).

    Returns:
    -------
    Fleet_InstanceType: A static instance of the Fleet jitclass.
    """
    fuel_instances_by_name: Dict[str, Fuel_InstanceType] = {}
    fuels_raw_by_name: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for fuel_id, fuel_year_dict in fuels_imported_dict.items():
        any_year_data = next(iter(fuel_year_dict.values()))
        fuel_name = str(any_year_data["name"])
        fuel_instances_by_name[fuel_name] = construct_Fuel_object(fuel_id, fuel_year_dict, firstyear, finalyear)
        fuels_raw_by_name[fuel_name] = fuel_year_dict

    generators = TypedDict.empty(key_type=int64, value_type=Generator_InstanceType)
    for order, generator_id in enumerate(generators_imported_dict):
        generators[order] = construct_Generator_object(
            generator_id,
            generators_imported_dict[generator_id],
            fuel_instances_by_name,
            fuels_raw_by_name,
            nodes_object_dict,
            lines_object_dict,
            order,
            firstyear,
            finalyear,
        )

    storages = TypedDict.empty(key_type=int64, value_type=Storage_InstanceType)
    for order, storage_id in enumerate(storages_imported_dict):
        storages[order] = construct_Storage_object(
            storage_id,
            storages_imported_dict[storage_id],
            nodes_object_dict,
            lines_object_dict,
            order,
            firstyear,
            finalyear,
        )

    return Fleet(
        True,
        generators,
        storages,
    )
