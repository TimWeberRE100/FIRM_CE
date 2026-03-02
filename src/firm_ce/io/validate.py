import os
from typing import Any, Collection, Dict, List, Tuple

import numpy as np
import pandas as pd

from firm_ce.common.helpers import parse_comma_separated
from firm_ce.common.logging import get_logger
from firm_ce.io.file_manager import import_config_csvs
from firm_ce.common.constants import YEAR_ALL_STR, SCENARIOS_ALL_STR


def validate_range(val: Any, min_val: float, max_val: float = None, inclusive: bool = True) -> bool:
    """
    Check that a value falls within a numeric range.

    Parameters:
    -------
    val (any): The value to check. Will be cast to float.
    min_val (float): The lower bound of the acceptable range.
    max_val (float): The upper bound of the acceptable range, or None for no upper bound.
    inclusive (bool): If True, the bounds are included (<=); otherwise excluded (<).

    Returns:
    -------
    bool: True if val is a valid float within the specified range, False otherwise.
    """
    try:
        val = float(val)
        if inclusive:
            return min_val <= val <= max_val if max_val is not None else min_val <= val
        else:
            return min_val < val < max_val if max_val is not None else min_val < val
    except TypeError, ValueError:
        return False


def validate_positive_int(val: Any) -> bool:
    """
    Check that a value can be cast to a positive integer.

    Parameters:
    -------
    val (any): The value to check.

    Returns:
    -------
    bool: True if val is a positive integer (> 0), False otherwise.
    """
    try:
        return int(val) > 0
    except TypeError, ValueError:
        return False


def validate_enum(val: Any, options: Collection) -> bool:
    """
    Check that a value is one of an allowed set of options.

    Parameters:
    -------
    val (Any): The value to check.
    options (Collection): A collection of acceptable values.

    Returns:
    -------
    bool: True if val is in options, False otherwise.
    """
    return val in options


def parse_list(val: str | float, lower: bool = True) -> list:
    """
    Parse a comma-separated string into a list, returning an empty list for NaN.

    Parameters:
    -------
    val (str | float): A comma-separated string, or a NaN float (e.g. from an empty CSV cell).
    lower (bool): If True, converts each item to lowercase before returning. Defaults to True.

    Returns:
    -------
    list: A list of stripped string tokens, or an empty list if val is NaN.
    """
    return parse_comma_separated(val, lower) if not is_nan(val) else []


def is_nan(val: str | float) -> bool:
    """
    Return True if val is a float NaN (e.g. an empty CSV cell read by Pandas).

    Parameters:
    -------
    val (str | float): The value to test.

    Returns:
    -------
    bool: True if val is float('nan'), False otherwise.
    """
    return isinstance(val, float) and np.isnan(val)


def get_applicable_scenarios(
    any_year_data: Dict[str, Any],
    scenarios_list: List[str],
    asset_id: int,
    asset_type: str,
) -> List[str]:
    """
    Return the scenarios this asset applies to. Unrecognised scenario names log a warning.

    Parameters:
    -------
    any_year_data (Dict[str, Any]): Any single year's data dict for the asset (used to read the
        'scenarios' field).
    scenarios_list (List[str]): Full list of valid scenario names from `scenarios.csv`.
    asset_id (int): Asset row id, used in warning messages.
    asset_type (str): Human-readable asset type label used in warning messages (e.g., 'generator').

    Returns:
    -------
    List[str]: Scenario names to which this asset should be assigned.
    """
    asset_scenarios = parse_list(any_year_data.get("scenarios"))
    if asset_scenarios == [SCENARIOS_ALL_STR]:
        return list(scenarios_list)
    applicable_scenarios = []
    for scenario in asset_scenarios:
        if scenario in scenarios_list:
            applicable_scenarios.append(scenario)
        else:
            get_logger().warning(
                "scenario '%s' for %s.id %s not defined in scenarios.csv",
                scenario,
                asset_type,
                asset_id,
            )
    return applicable_scenarios


def check_line_in_scenario(
    flag: bool,
    scenario: str,
    static_data: Dict[str, Any],
    scenario_lines: Dict[str, List[str]],
    scenario_minor_lines: Dict[str, List[str]],
    scenario_nodes: Dict[str, List[str]],
) -> bool:
    """
    Appends the line name to scenario_lines (and scenario_minor_lines if either endpoint is NaN),
    then checks that each non-NaN endpoint node exists in the scenario's node list. node_start and
    node_end are static fields read from `lines.csv` and do not vary by year.

    Parameters:
    -------
    flag (bool): Current validation flag. Returned unchanged unless a new error is found.
    scenario (str): The scenario name this line is being checked under.
    static_data (Dict[str, Any]): Static attribute dict for the line (any year's dict, since
        node_start and node_end are added as static fields from `lines.csv`).
    scenario_lines (Dict[str, List[str]]): Mapping of scenario name to line names.
    scenario_minor_lines (Dict[str, List[str]]): Mapping of scenario name to minor line names.
    scenario_nodes (Dict[str, List[str]]): Mapping of scenario name to valid node names.

    Returns:
    -------
    bool: Updated validation flag. False if any node endpoint check fails.

    Side-effects:
    -------
    Appends the line name to scenario_lines[scenario]. If either endpoint is NaN, also appends the
    line name to scenario_minor_lines[scenario].
    """
    name = static_data["name"]
    scenario_lines[scenario].append(name)

    if any(is_nan(static_data.get(n)) for n in ["node_start", "node_end"]):
        scenario_minor_lines[scenario].append(name)

    for endpoint in ["node_start", "node_end"]:
        node_val = static_data.get(endpoint)
        if (node_val not in scenario_nodes[scenario]) and not is_nan(node_val):
            get_logger().error(
                "'%s' %s for line %s is not defined in scenario %s",
                endpoint,
                node_val,
                name,
                scenario,
            )
            return False

    return flag


def check_generator_in_scenario(
    flag: bool,
    scenario: str,
    any_year_data: Dict[str, Any],
    scenario_generators: Dict[str, List[str]],
    scenario_baseload: Dict[str, List[str]],
    scenario_nodes: Dict[str, List[str]],
    scenario_fuels: Dict[str, List[str]],
    scenario_lines: Dict[str, List[str]],
) -> bool:
    """
    Append the generator name to the scenario's generator list, and if it is a baseload unit,
    also append it to the scenario's baseload list.

    Checks for duplicate generator names and validates that the referenced node, fuel, and line
    exist within the scenario.

    Parameters:
    -------
    flag (bool): Current validation flag. Returned unchanged unless a new error is found.
    scenario (str): The scenario name this generator is being registered under.
    any_year_data (Dict[str, Any]): Any single year's attribute dict for the generator (name,
        unit_type, node, fuel, and line do not vary by year).
    scenario_generators (Dict[str, List[str]]): Mapping of scenario name to registered generator
        names; mutated in place.
    scenario_baseload (Dict[str, List[str]]): Mapping of scenario name to baseload generator names;
        mutated in place.
    scenario_nodes (Dict[str, List[str]]): Mapping of scenario name to valid node names.
    scenario_fuels (Dict[str, List[str]]): Mapping of scenario name to valid fuel names.
    scenario_lines (Dict[str, List[str]]): Mapping of scenario name to valid line names.

    Returns:
    -------
    bool: Updated validation flag; False if any duplicate name or cross-reference error is found.

    Side-effects:
    -------
    Appends the generator name to scenario_generators[scenario]. If unit_type is 'baseload',
    also appends the name to scenario_baseload[scenario].
    """
    name = any_year_data["name"]

    if name in scenario_generators[scenario]:
        get_logger().error("Duplicate generator name '%s' in scenario %s", name, scenario)
        flag = False
    else:
        scenario_generators[scenario].append(name)

    if any_year_data["unit_type"] == "baseload":
        scenario_baseload[scenario].append(name)

    if any_year_data["node"] not in scenario_nodes[scenario]:
        get_logger().error(
            "'node' %s for generator %s is not defined in scenario %s",
            any_year_data["node"],
            name,
            scenario,
        )
        flag = False

    if any_year_data["fuel"] not in scenario_fuels[scenario]:
        get_logger().error(
            "'fuel' %s for generator %s is not defined in scenario %s",
            any_year_data["fuel"],
            name,
            scenario,
        )
        flag = False

    if any_year_data["line"] not in scenario_lines[scenario]:
        get_logger().error(
            "'line' %s for generator %s is not defined in scenario %s",
            any_year_data["line"],
            name,
            scenario,
        )
        flag = False

    return flag


def check_storage_in_scenario(
    flag: bool,
    scenario: str,
    any_year_data: Dict[str, Any],
    scenario_storages: Dict[str, List[str]],
    scenario_nodes: Dict[str, List[str]],
    scenario_lines: Dict[str, List[str]],
) -> bool:
    """
    Append the storage name to the scenario's storage list.

    Checks for duplicate storage names, then validates that the referenced node and line exist
    within the scenario.

    Parameters:
    -------
    flag (bool): Current validation flag. Returned unchanged unless a new error is found.
    scenario (str): The scenario name this storage is being registered under.
    any_year_data (Dict[str, Any]): Any single year's attribute dict for the storage (name, node,
        and line do not vary by year).
    scenario_storages (Dict[str, List[str]]): Mapping of scenario name to registered storage names;
        mutated in place.
    scenario_nodes (Dict[str, List[str]]): Mapping of scenario name to valid node names.
    scenario_lines (Dict[str, List[str]]): Mapping of scenario name to valid line names.

    Returns:
    -------
    bool: Updated validation flag. False if any duplicate name or cross-reference error is found.
    """
    name = any_year_data["name"]

    if name in scenario_storages[scenario]:
        get_logger().error("Duplicate storage name '%s' in scenario %s", name, scenario)
        flag = False
    else:
        scenario_storages[scenario].append(name)

    if any_year_data["node"] not in scenario_nodes[scenario]:
        get_logger().error(
            "'node' %s for storage %s is not defined in scenario %s",
            any_year_data["node"],
            name,
            scenario,
        )
        flag = False

    if any_year_data["line"] not in scenario_lines[scenario]:
        get_logger().error(
            "'line' %s for storage %s is not defined in scenario %s",
            any_year_data["line"],
            name,
            scenario,
        )
        flag = False

    return flag


def log_input_validation_result(input_name: str, flag: bool) -> None:
    """
    Log a per-input CSV validation outcome at INFO level on success or ERROR level on failure.

    Parameters:
    -------
    input_name (str): Name of the config file being reported (e.g. 'scenarios.csv').
    flag (bool): True if validation passed, False if any error was found.

    Returns:
    -------
    None.
    """
    if flag:
        get_logger().info("%s validated!", input_name)
    else:
        get_logger().error("%s contains errors.", input_name)


def validate_time_columns(df: pd.DataFrame, file_path: str) -> bool:
    """
    Check that a time-series DataFrame contains the four required time columns with valid values.

    Validates the presence of Year, Month, Day, and Interval columns, and checks that each
    contains values within its expected range. Returns False immediately if any column is absent.

    Parameters:
    -------
    df (pd.DataFrame): The loaded time-series DataFrame.
    file_path (str): Path to the source file, used in error messages.

    Returns:
    -------
    bool: False if any required column is missing or contains out-of-range values, True otherwise.
    """
    required_time_cols = ["Year", "Month", "Day", "Interval"]
    flag = True

    for col in required_time_cols:
        if col not in df.columns:
            get_logger().error("File '%s' is missing required column '%s'", file_path, col)
            flag = False

    if not flag:
        return False

    try:
        if not (df["Year"] >= 1).all():
            get_logger().error("'Year' values in '%s' must be positive integers", file_path)
            flag = False
    except Exception:
        get_logger().error("'Year' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not df["Month"].isin(range(1, 13)).all():
            get_logger().error("'Month' values in '%s' must be in range [1, 12]", file_path)
            flag = False
    except Exception:
        get_logger().error("'Month' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not df["Day"].between(1, 31).all():
            get_logger().error("'Day' values in '%s' must be in range [1, 31]", file_path)
            flag = False
    except Exception:
        get_logger().error("'Day' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not (df["Interval"] >= 1).all():
            get_logger().error("'Interval' values in '%s' must be >= 1", file_path)
            flag = False
    except Exception:
        get_logger().error("'Interval' column in '%s' contains non-numeric values", file_path)
        flag = False

    return flag


def validate_model_config(config_dict: Dict[int, Dict[str, Any]]) -> bool:
    """
    Validate all entries in `config.csv` against their expected types and ranges.

    Each known configuration key is checked with a dedicated validator. Unknown
    keys produce a warning. Keys with a None validator (e.g. model_name) are
    accepted without range checking.

    Parameters:
    -------
    config_dict (Dict[int, Dict[str, Any]]): Mapping of row index to {"name": ..., "value": ...} dicts
        as loaded from `config.csv`.

    Returns:
    -------
    bool: True if all known configuration values are valid, False if any fail.
    """
    flag = True
    validators = {
        "mutation": lambda v: validate_range(v, 0, 2, inclusive=False),
        "iterations": validate_positive_int,
        "population": validate_positive_int,
        "recombination": lambda v: validate_range(v, 0, 1),
        "type": lambda v: validate_enum(
            v,
            ["single_time", "capacity_expansion", "near_optimum", "midpoint_explore"],
        ),
        "model_name": None,
        "near_optimal_tol": lambda v: validate_range(v, 0, 1),
        "midpoint_count": validate_positive_int,
        "balancing_type": lambda v: validate_enum(v, ["simple", "full"]),
        "simple_blocks_per_day": validate_positive_int,
        "fixed_costs_threshold": lambda v: validate_range(v, 0),
    }

    for item in config_dict.values():
        name = item.get("name")
        value = item.get("value")

        if name not in validators:
            get_logger().warning("Unknown configuration name %s", name)
            continue

        if not validators[name]:
            continue

        try:
            if not validators[name](value):
                get_logger().error("Invalid value for '%s': %s", name, value)
                flag = False
        except Exception as e:
            get_logger().exception("Exception during validation of '%s': %s", name, e)
            flag = False

    return flag


def validate_scenarios(scenarios_dict: Dict[int, Dict[str, Any]]) -> Tuple[List[str], bool]:
    """
    Validate all rows in `scenarios.csv` and extract the list of scenario names.

    Checks for duplicate scenario names, valid numeric ranges for resolution and
    allowance, and that firstyear <= finalyear.

    Parameters:
    -------
    scenarios_dict (Dict[int, Dict[str, Any]]): Mapping of row index to scenario attribute dicts as loaded
        from scenarios.csv.

    Returns:
    -------
    Tuple[List[str], bool]: A 2-tuple (scenarios_list, flag) where:
        scenarios_list (List[str]) is an ordered list of scenario name strings;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenarios_list = []
    firstyear = finalyear = None

    for item in scenarios_dict.values():
        name = item["scenario_name"]
        if name in scenarios_list:
            get_logger().error("Duplicate scenario name '%s'", name)
            flag = False
        scenarios_list.append(name)

        if not validate_range(item["resolution"], 0):
            get_logger().error("'resolution' must be float greater than 0")
            flag = False

        if not validate_range(item["allowance"], 0, 1):
            get_logger().error("'allowance' must be float in range [0,1]")
            flag = False

        try:
            fy = int(item["firstyear"])
            ly = int(item["finalyear"])
            firstyear = fy if firstyear is None else min(firstyear, fy)
            finalyear = ly if finalyear is None else max(finalyear, ly)
        except ValueError:
            get_logger().error("'firstyear' and 'finalyear' must be integers")
            flag = False

    if firstyear is not None and finalyear is not None and firstyear > finalyear:
        get_logger().error("'firstyear' must be less than or equal to 'finalyear'")
        flag = False

    return scenarios_list, flag


def validate_nodes(
    nodes_dict: Dict[int, Dict[str, Any]], scenarios_list: List[str]
) -> Tuple[Dict[str, List[str]], bool]:
    """
    Validate all rows in `nodes.csv` and build a per-scenario node index.

    Checks for duplicate node names and that any scenario references exist in
    `scenarios.csv`. Errors if a referenced scenario is not defined.

    Parameters:
    -------
    nodes_dict (Dict[int, Dict[str, Any]]): Mapping of node id to attribute dicts as loaded
        from `nodes.csv`.
    scenarios_list (List[str]): List of valid scenario names from validate_scenarios.

    Returns:
    -------
    Tuple[Dict[str, List[str]], bool]: A 2-tuple (scenario_nodes, flag) where:
        scenario_nodes (Dict[str, List[str]]) maps scenario name to a list of node name strings;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_nodes = {s: [] for s in scenarios_list}
    node_names = []

    for _asset_id, node_data in nodes_dict.items():
        name = node_data.get("name")
        if name in node_names:
            get_logger().error("Duplicate node name '%s'", name)
            flag = False
        node_names.append(name)

        scenarios = parse_comma_separated(node_data.get("scenarios"))
        if scenarios == [SCENARIOS_ALL_STR]:
            for scenario in scenario_nodes.keys():
                scenario_nodes[scenario].append(name)
        else:
            for scenario in scenarios:
                if scenario not in scenarios_list:
                    get_logger().error("Scenario '%s' of node '%s' not in scenarios.csv", scenario, name)
                    flag = False
                scenario_nodes[scenario].append(name)

    return scenario_nodes, flag


def validate_fuels(
    fuels_dict: Dict[int, Dict[int, Dict[str, Any]]], scenarios_list: List[str]
) -> Tuple[Dict[str, List[str]], bool]:
    """
    Validate all rows in `fuels.csv` / `fuels_multiyear.csv` and build a per-scenario fuel index.

    Checks that emissions and cost are non-negative for each year, and warns if a fuel
    references a scenario not defined in `scenarios.csv`.

    Parameters:
    -------
    fuels_dict (Dict[int, Dict[int, Dict[str, Any]]]): Mapping of fuel id to year-keyed dicts as
        loaded and merged from `fuels.csv` and `fuels_multiyear.csv`. Each year dict maps column names to values.
    scenarios_list (List[str]): List of valid scenario names from validate_scenarios.

    Returns:
    -------
    Tuple[Dict[str, List[str]], bool]: A 2-tuple (scenario_fuels, flag) where:
        scenario_fuels (Dict[str, List[str]]) maps scenario name to a list of fuel name strings;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_fuels = {scenario: [] for scenario in scenarios_list}

    for asset_id, year_dict in fuels_dict.items():
        for year, item in year_dict.items():
            if not validate_range(item["emissions"], 0):
                get_logger().error(
                    "'emissions' must be float greater than or equal to 0 (id=%s, year=%s)", asset_id, year
                )
                flag = False

            if not validate_range(item["cost"], 0):
                get_logger().error("'cost' must be float greater than or equal to 0 (id=%s, year=%s)", asset_id, year)
                flag = False

        any_year_data = next(iter(year_dict.values()))
        for scenario in get_applicable_scenarios(any_year_data, scenarios_list, asset_id, "fuel"):
            scenario_fuels[scenario].append(any_year_data["name"])

    return scenario_fuels, flag


def validate_lines(
    lines_dict: Dict[int, Dict[int, Dict[str, Any]]],
    scenarios_list: List[str],
    scenario_nodes: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], bool]:
    """
    Validate all rows in `lines.csv` / `lines_multiyear.csv` and build per-scenario line indexes.

    Static fields (node_start, node_end, length, lifetime) are validated once per asset from
    `lines.csv`. Year-varying fields (capex, fom, vom, discount_rate, loss_factor, initial_capacity,
    max_build, min_build) are validated per year from `lines_multiyear.csv`. Also checks that
    min_build <= max_build and classifies lines without both endpoints (minor lines used to connect
    generators / storages to transmission backbone) separately.

    Parameters:
    -------
    lines_dict (Dict[int, Dict[int, Dict[str, Any]]]): Mapping of line id to year-keyed dicts as
        loaded and merged from lines.csv and lines_multiyear.csv. Each year dict maps column names to values.
    scenarios_list (List[str]): List of valid scenario names from validate_scenarios.
    scenario_nodes (Dict[str, List[str]]): Mapping of scenario name to a list of valid node names.

    Returns:
    -------
    Tuple[Dict[str, List[str]], Dict[str, List[str]], bool]: A 3-tuple (scenario_lines, scenario_minor_lines, flag) where:
        scenario_lines (Dict[str, List[str]]) maps scenario name to a list of line name strings;
        scenario_minor_lines (Dict[str, List[str]]) maps scenario name to a list of line names that are
        missing at least one endpoint node;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_lines = {s: [] for s in scenarios_list}
    scenario_minor_lines = {s: [] for s in scenarios_list}

    static_int_fields = {"length": int, "lifetime": int}
    year_numeric_fields = {
        "capex": float,
        "transformer_capex": float,
        "fom": float,
        "vom": float,
        "discount_rate": float,
        "loss_factor": float,
        "initial_capacity": float,
        "max_build": float,
        "min_build": float,
    }

    for asset_id, year_dict in lines_dict.items():
        any_year_data = next(iter(year_dict.values()))

        for field, cast in static_int_fields.items():
            try:
                val = cast(any_year_data[field])
                if val < 0:
                    raise ValueError
            except TypeError, ValueError:
                get_logger().error(
                    "'%s' must be a valid %s >= 0 (id=%s)",
                    field,
                    cast.__name__,
                    asset_id,
                )
                flag = False

        for year, item in year_dict.items():
            fields_valid = True
            for field, cast in year_numeric_fields.items():
                try:
                    val = cast(item[field])
                    if field == "discount_rate":
                        if not (0 <= val <= 1):
                            raise ValueError
                    elif field == "loss_factor":
                        if not (0 <= val < 1):
                            raise ValueError
                    else:
                        if val < 0:
                            raise ValueError
                except ValueError:
                    get_logger().error(
                        "'%s' must be a valid %s in appropriate range (id=%s, year=%s)",
                        field,
                        cast.__name__,
                        asset_id,
                        year,
                    )
                    flag = False
                    if field in ("min_build", "max_build"):
                        fields_valid = False

            if fields_valid:
                try:
                    if float(item["min_build"]) > float(item["max_build"]):
                        get_logger().error(
                            "'min_build' must be less than or equal to 'max_build' (id=%s, year=%s)", asset_id, year
                        )
                        flag = False
                except TypeError, ValueError:
                    pass

        for scenario in get_applicable_scenarios(any_year_data, scenarios_list, asset_id, "line"):
            flag = check_line_in_scenario(
                flag, scenario, any_year_data, scenario_lines, scenario_minor_lines, scenario_nodes
            )

    return scenario_lines, scenario_minor_lines, flag


def validate_generators(
    generators_dict: Dict[int, Dict[int, Dict[str, Any]]],
    scenarios_list: List[str],
    scenario_fuels: Dict[str, List[str]],
    scenario_lines: Dict[str, List[str]],
    scenario_nodes: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], bool]:
    """
    Validate all rows in `generators.csv` / `generators_multiyear.csv` and build per-scenario generator indexes.

    Checks that lifetime is a non-negative integer (static field), then for each year checks
    non-negative numeric fields, discount_rate range, valid unit_type, and that min_build <= max_build.
    Also validates that the referenced node, fuel, and line exist within the relevant scenario.

    Parameters:
    -------
    generators_dict (Dict[int, Dict[int, Dict[str, Any]]]): Mapping of generator id to year-keyed dicts
        as loaded and merged from `generators.csv` and `generators_multiyear.csv`. Each year dict maps column names to values.
    scenarios_list (List[str]): List of valid scenario names.
    scenario_fuels (Dict[str, List[str]]): Mapping of scenario name to a list of valid fuel names.
    scenario_lines (Dict[str, List[str]]): Mapping of scenario name to a list of valid line names.
    scenario_nodes (Dict[str, List[str]]): Mapping of scenario name to a list of valid node names.

    Returns:
    -------
    Tuple[Dict[str, List[str]], Dict[str, List[str]], bool]: A 3-tuple (scenario_generators, scenario_baseload, flag) where:
        scenario_generators (Dict[str, List[str]]) maps scenario name to a list of generator names;
        scenario_baseload (Dict[str, List[str]]) maps scenario name to a list of baseload generator names;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_generators = {s: [] for s in scenarios_list}
    scenario_baseload = {s: [] for s in scenarios_list}

    for asset_id, year_dict in generators_dict.items():
        any_year_data = next(iter(year_dict.values()))

        try:
            if int(any_year_data["lifetime"]) <= 0:
                get_logger().error("'lifetime' must be int > 0 (id=%s)", asset_id)
                flag = False
        except TypeError, ValueError:
            get_logger().error("'lifetime' must be a valid integer (id=%s)", asset_id)
            flag = False

        for year, item in year_dict.items():
            fields_valid = True
            for field in [
                "capex",
                "fom",
                "vom",
                "heat_rate_base",
                "heat_rate_incr",
                "initial_capacity",
                "max_build",
                "min_build",
            ]:
                if not validate_range(item[field], 0):
                    get_logger().error(
                        "'%s' must be float greater than or equal to 0 (id=%s, year=%s)", field, asset_id, year
                    )
                    flag = False
                    if field in ("min_build", "max_build"):
                        fields_valid = False

            if not validate_range(item["discount_rate"], 0, 1):
                get_logger().error("'discount_rate' must be float in range [0,1] (id=%s, year=%s)", asset_id, year)
                flag = False

            if not validate_enum(item["unit_type"], ["solar", "wind", "flexible", "baseload"]):
                get_logger().error(
                    "'unit_type' must be one of ['solar', 'wind', 'flexible', 'baseload'] (id=%s, year=%s)",
                    asset_id,
                    year,
                )
                flag = False

            if fields_valid:
                try:
                    if float(item["min_build"]) > float(item["max_build"]):
                        get_logger().error(
                            "'min_build' must be less than or equal to 'max_build' (id=%s, year=%s)", asset_id, year
                        )
                        flag = False
                except TypeError, ValueError:
                    pass

        for scenario in get_applicable_scenarios(any_year_data, scenarios_list, asset_id, "generator"):
            flag = check_generator_in_scenario(
                flag,
                scenario,
                any_year_data,
                scenario_generators,
                scenario_baseload,
                scenario_nodes,
                scenario_fuels,
                scenario_lines,
            )

    return scenario_generators, scenario_baseload, flag


def validate_storages(
    storages_dict: Dict[int, Dict[int, Dict[str, Any]]],
    scenarios_list: List[str],
    scenario_nodes: Dict[str, List[str]],
    scenario_lines: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], bool]:
    """
    Validate all rows in `storages.csv` / `storages_multiyear.csv` and build a per-scenario storage index.

    Checks that discount_rate is in [0, 1] (static field), then for each year checks non-negative
    capacity and cost fields, that min/max build pairs are ordered correctly, that lifetime and
    duration are non-negative integers, and that efficiency values are in [0, 1]. Also validates
    that the referenced node and line exist within the relevant scenario.

    Parameters:
    -------
    storages_dict (Dict[int, Dict[int, Dict[str, Any]]]): Mapping of storage id to year-keyed dicts
        as loaded and merged from storages.csv and storages_multiyear.csv. Each year dict maps column names to values.
    scenarios_list (List[str]): List of valid scenario names.
    scenario_nodes (Dict[str, List[str]]): Mapping of scenario name to a list of valid node names.
    scenario_lines (Dict[str, List[str]]): Mapping of scenario name to a list of valid line names.

    Returns:
    -------
    Tuple[Dict[str, List[str]], bool]: A 2-tuple (scenario_storages, flag) where:
        scenario_storages (Dict[str, List[str]]) maps scenario name to a list of storage names;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_storages = {s: [] for s in scenarios_list}

    for asset_id, year_dict in storages_dict.items():
        any_year_data = next(iter(year_dict.values()))

        if not validate_range(any_year_data["discount_rate"], 0, 1):
            get_logger().error("'discount_rate' must be float in [0,1] (id=%s)", asset_id)
            flag = False

        for year, item in year_dict.items():
            build_fields_valid = True
            for field in [
                "capex_p",
                "capex_e",
                "fom",
                "vom",
                "initial_power_capacity",
                "initial_energy_capacity",
                "max_build_p",
                "min_build_p",
                "max_build_e",
                "min_build_e",
            ]:
                if not validate_range(item[field], 0):
                    get_logger().error("'%s' must be float >= 0 (id=%s, year=%s)", field, asset_id, year)
                    flag = False
                    if field in ("min_build_p", "max_build_p", "min_build_e", "max_build_e"):
                        build_fields_valid = False

            if build_fields_valid:
                for min_field, max_field in [("min_build_p", "max_build_p"), ("min_build_e", "max_build_e")]:
                    try:
                        if float(item[min_field]) > float(item[max_field]):
                            get_logger().error(
                                "'%s' must be <= '%s' (id=%s, year=%s)", min_field, max_field, asset_id, year
                            )
                            flag = False
                    except TypeError, ValueError:
                        pass

            try:
                if int(item["lifetime"]) <= 0:
                    get_logger().error("'lifetime' must be int > 0 (id=%s, year=%s)", asset_id, year)
                    flag = False
            except TypeError, ValueError:
                get_logger().error("'lifetime' must be a valid integer (id=%s, year=%s)", asset_id, year)
                flag = False

            try:
                if int(item["duration"]) < 0:
                    get_logger().error("'duration' must be int >= 0 (id=%s, year=%s)", asset_id, year)
                    flag = False
            except TypeError, ValueError:
                get_logger().error("'duration' must be a valid integer (id=%s, year=%s)", asset_id, year)
                flag = False

            for efficiency in ["charge_efficiency", "discharge_efficiency"]:
                if not validate_range(item[efficiency], 0, 1):
                    get_logger().error("'%s' must be float in [0,1] (id=%s, year=%s)", efficiency, asset_id, year)
                    flag = False

        for scenario in get_applicable_scenarios(any_year_data, scenarios_list, asset_id, "storage"):
            flag = check_storage_in_scenario(
                flag, scenario, any_year_data, scenario_storages, scenario_nodes, scenario_lines
            )

    return scenario_storages, flag


def validate_initial_guess(
    x0s_dict: Dict[int, Dict[str, Any]],
    scenarios_list: List[str],
    scenario_generators: Dict[str, List[str]],
    scenario_storages: Dict[str, List[str]],
    scenario_lines: Dict[str, List[str]],
    scenario_baseload: Dict[str, List[str]],
    scenario_minor_lines: Dict[str, List[str]],
) -> bool:
    """
    Validate all rows in `initial_guess.csv`.

    Checks that each x_0 vector has the correct length for its scenario (one value per
    generator, two per storage [power and energy] and one per major line), and that every
    scenario defined in `scenarios.csv` has a corresponding entry.

    Parameters:
    -------
    x0s_dict (Dict[int, Dict[str, Any]]): Mapping of row index to initial-guess attribute dicts as loaded
        from `initial_guess.csv`.
    scenarios_list (List[str]): List of valid scenario names.
    scenario_generators (Dict[str, List[str]]): Mapping of scenario name to a list of generator names.
    scenario_storages (Dict[str, List[str]]): Mapping of scenario name to a list of storage names.
    scenario_lines (Dict[str, List[str]]): Mapping of scenario name to a list of line names.
    scenario_baseload (Dict[str, List[str]]): Mapping of scenario name to a list of baseload generator names.
    scenario_minor_lines (Dict[str, List[str]]): Mapping of scenario name to a list of minor line names
        (lines without both endpoints).

    Returns:
    -------
    bool: True if all initial-guess entries are valid and complete, False otherwise.
    """
    flag = True
    initial_guess_scenarios = []

    for item in x0s_dict.values():
        scenario = item["scenario"]

        if scenario not in scenarios_list:
            get_logger().warning("scenario '%s' in initial_guess.csv not defined in scenarios.csv", scenario)
            continue

        initial_guess_scenarios.append(scenario)

        x0 = parse_list(item["x_0"])
        bound_length = len(
            scenario_generators[scenario]
            + scenario_storages[scenario]
            + scenario_storages[scenario]
            + scenario_lines[scenario]
        ) - len(scenario_minor_lines[scenario])

        if x0 and len(x0) != bound_length:
            get_logger().error(
                "Initial guess 'x_0' for scenario %s contains %d elements, expected %d",
                scenario,
                len(x0),
                bound_length,
            )
            flag = False

    for scenario in scenarios_list:
        if scenario not in initial_guess_scenarios:
            get_logger().error("scenario '%s' is defined in scenarios.csv but missing from initial_guess.csv", scenario)
            flag = False

    return flag


def get_config_type(config_dict: Dict[int, Dict[str, Any]]) -> str | None:
    """
    Extract the 'type' value from a `config.csv` dict.

    Parameters:
    -------
    config_dict (Dict[int, Dict[str, Any]]): Mapping of row index to config attribute dicts as
        loaded from `config.csv`.

    Returns:
    -------
    str | None: The value of the 'type' entry, or None if not present.
    """
    for record in config_dict.values():
        if record.get("name") == "type":
            return record.get("value")
    return None


def validate_multiyear_year_columns(multiyear_dict: Dict[int, Dict], config_type: str, filename: str) -> bool:
    """
    Validate that year column values in a multiyear config dict are consistent with config.type.

    For "single_time": every year key must be the YEAR_ALL_STR string. Integer year keys are
    not permitted because single-time models have no concept of per-year variation.
    For "capacity_expansion": year keys may be integers or YEAR_ALL_STR.

    Parameters:
    -------
    multiyear_dict (Dict[int, Dict]): Mapping of asset id to year-keyed dicts as returned by
        get_multiyear_data. Year keys are integers or the string YEAR_ALL_STR.
    config_type (str): The 'type' value from `config.csv` (e.g. "single_time", "capacity_expansion").
    filename (str): Name of the config file being validated, used in error messages.

    Returns:
    -------
    bool: True if all year values are valid for the given config type, False otherwise.
    """
    if config_type is None:
        return True

    flag = True
    for asset_id, year_dict in multiyear_dict.items():
        for year_key in year_dict.keys():
            if year_key == YEAR_ALL_STR:
                continue
            if config_type == "single_time":
                get_logger().error(
                    "Config type 'single_time' requires year='all' for all rows in %s (id=%s, got year=%s)",
                    filename,
                    asset_id,
                    year_key,
                )
                flag = False
    return flag


def validate_config(config_directory: str) -> bool:
    """
    Load all config CSVs from config_directory and validate every input CSV in order.

    This function must be called before constructing a ModelData instance. It reads the CSV
    files directly from disk so that validation errors can be caught and reported before any
    model objects are built. The logger must be initialised before calling this function.

    Parameters:
    -------
    config_directory (str): Path to the directory containing all config CSV files.

    Returns:
    -------
    bool: True if all configuration tables pass validation, False otherwise.
    """
    config_data = import_config_csvs(config_directory)

    config_flag = validate_model_config(config_data.get("config", {}))
    log_input_validation_result("config.csv", config_flag)

    config_type = get_config_type(config_data.get("config", {}))
    multiyear_files = {
        "generators": config_data.get("generators"),
        "fuels": config_data.get("fuels"),
        "lines": config_data.get("lines"),
        "storages": config_data.get("storages"),
    }
    for mf_name, mf_dict in multiyear_files.items():
        if not validate_multiyear_year_columns(mf_dict, config_type, mf_name + ".csv"):
            get_logger().error("%s.csv contains invalid year values for config type '%s'.", mf_name, config_type)
            config_flag = False

    scenarios_list, flag = validate_scenarios(config_data.get("scenarios", {}))
    log_input_validation_result("scenarios.csv", flag)
    config_flag = config_flag and flag

    scenario_nodes, flag = validate_nodes(config_data.get("nodes", {}), scenarios_list)
    log_input_validation_result("nodes.csv", flag)
    config_flag = config_flag and flag

    scenario_fuels, flag = validate_fuels(config_data.get("fuels", {}), scenarios_list)
    log_input_validation_result("fuels.csv / fuels_multiyear.csv", flag)
    config_flag = config_flag and flag

    scenario_lines, scenario_minor_lines, flag = validate_lines(
        config_data.get("lines", {}), scenarios_list, scenario_nodes
    )
    log_input_validation_result("lines.csv / lines_multiyear.csv", flag)
    config_flag = config_flag and flag

    scenario_generators, scenario_baseload, flag = validate_generators(
        config_data.get("generators", {}), scenarios_list, scenario_fuels, scenario_lines, scenario_nodes
    )
    log_input_validation_result("generators.csv / generators_multiyear.csv", flag)
    config_flag = config_flag and flag

    scenario_storages, flag = validate_storages(
        config_data.get("storages", {}), scenarios_list, scenario_nodes, scenario_lines
    )
    log_input_validation_result("storages.csv / storages_multiyear.csv", flag)
    config_flag = config_flag and flag

    flag = validate_initial_guess(
        config_data.get("initial_guess", {}),
        scenarios_list,
        scenario_generators,
        scenario_storages,
        scenario_lines,
        scenario_baseload,
        scenario_minor_lines,
    )
    log_input_validation_result("initial_guess.csv", flag)
    config_flag = config_flag and flag

    return config_flag


def validate_datafiles_config(
    scenario_filenames: List[str], scenario_datafile_types: List[str], datafiles_directory: str
) -> bool:
    """
    Check that all referenced data files exist on disk and have valid types.

    Parameters:
    -------
    scenario_filenames (List[str]): List of filenames expected to be present in
        datafiles_directory.
    scenario_datafile_types (List[str]): List of datafile type strings corresponding to each
        filename. Valid values are "demand", "generation", and "flexible_annual_limit".
    datafiles_directory (str): Absolute path to the directory containing data files.

    Returns:
    -------
    bool: True if all files exist and all types are valid, False otherwise.
    """
    valid_types = {"demand", "generation", "flexible_annual_limit"}
    all_filenames = set(os.listdir(datafiles_directory))
    flag = True

    for fn in scenario_filenames:
        if fn not in all_filenames:
            get_logger().error("Missing file data/%s", fn)
            flag = False

    for dtype in scenario_datafile_types:
        if not dtype or dtype not in valid_types:
            get_logger().error("Invalid or missing datafile_type '%s'", dtype)
            flag = False

    return flag


def validate_electricity(file_path: str, node_list: List[str]) -> bool:
    """
    Validate a demand profile CSV file.

    Checks that the required time columns are present and contain valid values, and that each
    expected node has a column.

    Parameters:
    -------
    file_path (str): Absolute path to the demand CSV file.
    node_list (List[str]): List of node names expected as columns in the file. If None, column
        name validation is skipped and only value ranges are checked.

    Returns:
    -------
    bool: True if the file is structurally valid and all demand values are non-negative,
        False otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    except Exception as e:
        get_logger().error("Could not read demand file '%s': %s", file_path, e)
        return False

    if not validate_time_columns(df, file_path):
        return False

    flag = True

    if node_list is not None:
        for node in node_list:
            if node not in df.columns:
                get_logger().error("Demand file '%s' is missing column for node '%s'", file_path, node)
                flag = False

    required_time_cols = ["Year", "Month", "Day", "Interval"]
    data_cols = [c for c in df.columns if c not in required_time_cols]
    for col in data_cols:
        if df[col].isna().any():
            get_logger().error("Column '%s' in demand file '%s' contains NaN values", col, file_path)
            flag = False
        elif not pd.api.types.is_numeric_dtype(df[col]):
            get_logger().error("Column '%s' in demand file '%s' contains non-numeric values", col, file_path)
            flag = False

    return flag


def validate_generation(file_path: str, solar_list: List[str], wind_list: List[str], baseload_list: List[str]) -> bool:
    """
    Validate a generation trace CSV file for solar, wind, or baseload units.

    Checks that the required time columns are present and contain valid values, that all
    data columns contain capacity factors in [0, 1] with no missing data, and warns if
    any data column does not correspond to a configured solar, wind, or baseload generator.

    Parameters:
    -------
    file_path (str): Absolute path to the generation trace CSV file.
    solar_list (List[str]): List of solar generator names for the scenario. If all three
        generator lists are None, column name validation is skipped.
    wind_list (List[str]): List of wind generator names for the scenario.
    baseload_list (List[str]): List of baseload generator names for the scenario.

    Returns:
    -------
    bool: True if the file is structurally valid and all capacity factors are in [0, 1],
        False otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    except Exception as e:
        get_logger().error("Could not read generation file '%s': %s", file_path, e)
        return False

    if not validate_time_columns(df, file_path):
        return False

    flag = True
    required_time_cols = ["Year", "Month", "Day", "Interval"]
    data_cols = [c for c in df.columns if c not in required_time_cols]

    if solar_list is not None and wind_list is not None and baseload_list is not None:
        known_generators = set(solar_list) | set(wind_list) | set(baseload_list)
        for col in data_cols:
            if col not in known_generators:
                get_logger().warning(
                    "Column '%s' in generation file '%s' does not match any configured generator",
                    col,
                    file_path,
                )

    for col in data_cols:
        if df[col].isna().any():
            get_logger().error("Column '%s' in generation file '%s' contains NaN values", col, file_path)
            flag = False
        elif not df[col].between(0, 1).all():
            get_logger().error(
                "Column '%s' in generation file '%s' contains values outside [0, 1]",
                col,
                file_path,
            )
            flag = False

    return flag


def validate_flexible_limits(file_path: str, flexible_list: List[str]) -> bool:
    """
    Validate a flexible annual generation limit CSV file.

    Checks that a Year column is present and contains valid values, that each expected
    flexible generator has a column, and that all limit values are non-negative with no
    missing data. Unlike demand and generation files, this file has one row per year
    rather than per time interval.

    Parameters:
    -------
    file_path (str): Absolute path to the flexible annual limit CSV file.
    flexible_list (List[str]): List of flexible generator names expected as columns in the file.
        If None, column name validation is skipped and only value ranges are checked.

    Returns:
    -------
    bool: True if the file is structurally valid and all limit values are non-negative,
        False otherwise.
    """
    flag = True

    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    except Exception as e:
        get_logger().error("Could not read flexible limits file '%s': %s", file_path, e)
        return False

    if "Year" not in df.columns:
        get_logger().error("Flexible limits file '%s' is missing required column 'Year'", file_path)
        return False

    try:
        if not (df["Year"] >= 1).all():
            get_logger().error("'Year' values in '%s' must be positive integers", file_path)
            flag = False
    except Exception:
        get_logger().error("'Year' column in '%s' contains non-numeric values", file_path)
        flag = False

    if flexible_list is not None:
        for gen in flexible_list:
            if gen not in df.columns:
                get_logger().error(
                    "Flexible limits file '%s' is missing column for generator '%s'",
                    file_path,
                    gen,
                )
                flag = False

    data_cols = [c for c in df.columns if c != "Year"]
    for col in data_cols:
        if df[col].isna().any():
            get_logger().error("Column '%s' in flexible limits file '%s' contains NaN values", col, file_path)
            flag = False
        elif (df[col] < 0).any():
            get_logger().error("Column '%s' in flexible limits file '%s' contains negative values", col, file_path)
            flag = False

    return flag


def validate_data(
    all_datafiles: Dict[int, Dict[str, Any]],
    scenario_name: str,
    datafiles_directory: str,
    node_list: List[str] = None,
    solar_list: List[str] = None,
    wind_list: List[str] = None,
    baseload_list: List[str] = None,
    flexible_list: List[str] = None,
) -> bool:
    """
    Validate the data files referenced by a specific scenario.

    Filters datafiles.csv to entries for scenario_name, checks that all referenced files
    exist on disk with valid types, then validates the content of each file by delegating
    to validate_electricity, validate_generation, or validate_flexible_limits depending
    on the file's datafile_type.

    The optional generator and node lists are used to cross-reference file columns against
    the scenario configuration. If omitted, only structural and value-range checks are
    performed.

    Parameters:
    -------
    all_datafiles (Dict[int, Dict[str, Any]]): Mapping of row index to datafile attribute dicts as loaded
        from datafiles.csv.
    scenario_name (str): The name of the scenario whose data files should be validated.
    datafiles_directory (str): Absolute path to the directory containing data files.
    node_list (List[str]): List of node names expected in demand files for this scenario.
        Defaults to None (skips column name validation).
    solar_list (List[str]): List of solar generator names for this scenario.
        Defaults to None (skips column name validation).
    wind_list (List[str]): List of wind generator names for this scenario.
        Defaults to None (skips column name validation).
    baseload_list (List[str]): List of baseload generator names for this scenario.
        Defaults to None (skips column name validation).
    flexible_list (List[str]): List of flexible generator names expected in annual limit files.
        Defaults to None (skips column name validation).

    Returns:
    -------
    bool: True if all data files for the scenario are valid, False otherwise.
    """
    flag = True
    scenario_filenames = []
    scenario_datafile_types = []
    scenario_files_by_type: Dict[str, List[str]] = {"demand": [], "generation": [], "flexible_annual_limit": []}

    for item in all_datafiles.values():
        scenario_list = parse_comma_separated(item["scenarios"])
        if scenario_name in scenario_list:
            filename = item["filename"]
            dtype = item["datafile_type"]
            scenario_filenames.append(filename)
            scenario_datafile_types.append(dtype)
            if dtype in scenario_files_by_type:
                scenario_files_by_type[dtype].append(filename)

    if not validate_datafiles_config(scenario_filenames, scenario_datafile_types, datafiles_directory):
        get_logger().error("datafiles.csv contains errors for scenario %s.", scenario_name)
        flag = False
    else:
        get_logger().info("datafiles.csv validated for scenario %s!", scenario_name)

    for filename in scenario_files_by_type["demand"]:
        file_path = os.path.join(datafiles_directory, filename)
        if not validate_electricity(file_path, node_list):
            get_logger().error("Demand file '%s' contains errors for scenario %s.", filename, scenario_name)
            flag = False
        else:
            get_logger().info("Demand file '%s' validated for scenario %s!", filename, scenario_name)

    for filename in scenario_files_by_type["generation"]:
        file_path = os.path.join(datafiles_directory, filename)
        if not validate_generation(file_path, solar_list, wind_list, baseload_list):
            get_logger().error("Generation file '%s' contains errors for scenario %s.", filename, scenario_name)
            flag = False
        else:
            get_logger().info("Generation file '%s' validated for scenario %s!", filename, scenario_name)

    for filename in scenario_files_by_type["flexible_annual_limit"]:
        file_path = os.path.join(datafiles_directory, filename)
        if not validate_flexible_limits(file_path, flexible_list):
            get_logger().error("Flexible limits file '%s' contains errors for scenario %s.", filename, scenario_name)
            flag = False
        else:
            get_logger().info("Flexible limits file '%s' validated for scenario %s!", filename, scenario_name)

    return flag
