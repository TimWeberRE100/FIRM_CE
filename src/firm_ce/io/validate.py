import os

from typing import Any, Collection
from logging import Logger

import numpy as np
import pandas as pd

from firm_ce.common.helpers import parse_comma_separated
from firm_ce.common.logging import init_model_logger
from firm_ce.io.file_manager import import_config_csvs


class ModelData:
    """
    Container for all model configuration data loaded from CSV files.
    """

    def __init__(self, config_directory: str, logging_flag: bool) -> None:
        """
        Load configuration CSVs and initialise the model logger.

        Parameters:
        -------
        config_directory (str): Path to the directory containing all config CSV files.
        logging_flag (bool): Whether to enable file-based logging.
        """
        self.config_directory = config_directory

        self.config_data = import_config_csvs(config_directory=config_directory)

        model_name = self.get_model_name()

        self.logger, self.results_dir = init_model_logger(model_name, logging_flag)

        self.scenarios = self.config_data.get("scenarios")
        self.generators = self.config_data.get("generators")
        self.fuels = self.config_data.get("fuels")
        self.lines = self.config_data.get("lines")
        self.storages = self.config_data.get("storages")
        self.config = self.config_data.get("config")
        self.x0s = self.config_data.get("initial_guess")
        self.datafiles = self.config_data.get("datafiles")

    def validate(self) -> bool:
        """
        Validate all loaded configuration data.

        Returns:
        -------
        bool: True if all configuration dictionaries are valid, False otherwise.
        """
        return validate_config(self)

    def get_model_name(self) -> str:
        """
        Extract the model name from the config dictionary.

        Returns:
        -------
        str: The value of the model_name entry in config.csv, or "Model" if no such entry exists.
        """
        model_name = None

        if "config" in self.config_data:
            for record in self.config_data["config"].values():
                if record.get("name") == "model_name":
                    model_name = record.get("value")
                    break

        if model_name is None:
            model_name = "Model"

        return model_name


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
    except (TypeError, ValueError):
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
    except (TypeError, ValueError):
        return False


def validate_enum(val: Any, options: Collection) -> bool:
    """
    Check that a value is one of an allowed set of options.

    Parameters:
    -------
    val (any): The value to check.
    options (collection): A collection of acceptable values.

    Returns:
    -------
    bool: True if val is in options, False otherwise.
    """
    return val in options


def parse_list(val: str | float) -> list:
    """
    Parse a comma-separated string into a list, returning an empty list for NaN.

    Parameters:
    -------
    val (str | float): A comma-separated string, or a NaN float (e.g. from an empty CSV cell).

    Returns:
    -------
    list: A list of stripped string tokens, or an empty list if val is NaN.
    """
    return parse_comma_separated(val) if not is_nan(val) else []


def is_nan(val: str | float) -> bool:
    """
    Return True if val is a float NaN (e.g. an empty CSV cell read by pandas).

    Parameters:
    -------
    val (str | float): The value to test.

    Returns:
    -------
    bool: True if val is float('nan'), False otherwise.
    """
    return isinstance(val, float) and np.isnan(val)


def validate_model_config(config_dict: dict, model_logger: Logger) -> bool:
    """
    Validate all entries in config.csv against their expected types and ranges.

    Each known configuration key is checked with a dedicated validator. Unknown
    keys produce a warning; keys with a None validator (e.g. model_name) are
    accepted without range checking.

    Parameters:
    -------
    config_dict (dict): Mapping of row index to {"name": ..., "value": ...} dicts
        as loaded from config.csv.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

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
        "balancing_type": lambda v: validate_enum(
            v,
            ["simple", "full"],
        ),
        "simple_blocks_per_day": validate_positive_int,
        "fixed_costs_threshold": lambda v: validate_range(v, 0),
    }

    for item in config_dict.values():
        name = item.get("name")
        value = item.get("value")

        if name not in validators:
            model_logger.warning("Unknown configuration name %s", name)
            continue

        if not validators[name]:
            continue

        try:
            if not validators[name](value):
                model_logger.error("Invalid value for '%s': %s", name, value)
                flag = False
        except Exception as e:
            model_logger.exception("Exception during validation of '%s': %s", name, e)
            flag = False

    return flag


def validate_scenarios(scenarios_dict: dict, model_logger: Logger) -> tuple:
    """
    Validate all rows in scenarios.csv and extract per-scenario node/line lists.

    Checks for duplicate scenario names, valid numeric ranges for resolution and
    allowance, and that firstyear <= finalyear.

    Parameters:
    -------
    scenarios_dict (dict): Mapping of row index to scenario attribute dicts as loaded
        from scenarios.csv.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    tuple: A 4-tuple (scenarios_list, scenario_nodes, scenario_lines, flag) where:
        scenarios_list (list) is an ordered list of scenario name strings;
        scenario_nodes (dict) maps scenario name to a list of node names;
        scenario_lines (dict) maps scenario name to a list of line names sourced from
        the lines column of scenarios.csv;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenarios_list = []
    scenario_nodes = {}
    scenario_lines = {}
    firstyear = finalyear = None

    for item in scenarios_dict.values():
        name = item["scenario_name"]
        if name in scenarios_list:
            model_logger.error("Duplicate scenario name '%s'", name)
            flag = False
        scenarios_list.append(name)

        if not validate_range(item["resolution"], 0):
            model_logger.error("'resolution' must be float greater than 0")
            flag = False

        if not validate_range(item["allowance"], 0, 1):
            model_logger.error("'allowance' must be float in range [0,1]")
            flag = False

        try:
            fy = int(item["firstyear"])
            ly = int(item["finalyear"])
            firstyear = fy if firstyear is None else min(firstyear, fy)
            finalyear = ly if finalyear is None else max(finalyear, ly)
        except ValueError:
            model_logger.error("'firstyear' and 'finalyear' must be integers")
            flag = False

        scenario_nodes[name] = parse_list(item.get("nodes"))
        scenario_lines[name] = parse_list(item.get("lines"))

    if firstyear is not None and finalyear is not None and firstyear > finalyear:
        model_logger.error("'firstyear' must be less than or equal to 'finalyear'")
        flag = False

    return scenarios_list, scenario_nodes, scenario_lines, flag


def validate_fuels(fuels_dict: dict, scenarios_list: list, model_logger: Logger) -> tuple:
    """
    Validate all rows in fuels.csv and build a per-scenario fuel index.

    Checks that emissions and cost are non-negative, and warns if a fuel references
    a scenario not defined in scenarios.csv.

    Parameters:
    -------
    fuels_dict (dict): Mapping of row index to fuel attribute dicts as loaded from
        fuels.csv.
    scenarios_list (list): List of valid scenario names from validate_scenarios.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    tuple: A 2-tuple (scenario_fuels, flag) where:
        scenario_fuels (dict) maps scenario name to a list of fuel name strings;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_fuels = {scenario: [] for scenario in scenarios_list}

    for idx, item in fuels_dict.items():
        if not validate_range(item["emissions"], 0):
            model_logger.error("'emissions' must be float greater than or equal to 0")
            flag = False

        if not validate_range(item["cost"], 0):
            model_logger.error("'cost' must be float greater than or equal to 0")
            flag = False

        for scenario in parse_list(item.get("scenarios")):
            if scenario in scenarios_list:
                scenario_fuels[scenario].append(item["name"])
            else:
                model_logger.warning("'scenario' %s for fuel.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_fuels, flag


def validate_lines(lines_dict: dict, scenarios_list: list, scenario_nodes: dict, model_logger: Logger) -> tuple:
    """
    Validate all rows in lines.csv and build per-scenario line indexes.

    Checks numeric field types and ranges, that min_build <= max_build, that referenced
    nodes exist in the relevant scenario, and classifies lines without both endpoints
    (minor lines) separately.

    Parameters:
    -------
    lines_dict (dict): Mapping of row index to line attribute dicts as loaded from
        lines.csv.
    scenarios_list (list): List of valid scenario names from validate_scenarios.
    scenario_nodes (dict): Mapping of scenario name to a list of valid node names.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    tuple: A 3-tuple (scenario_lines, scenario_minor_lines, flag) where:
        scenario_lines (dict) maps scenario name to a list of line name strings;
        scenario_minor_lines (dict) maps scenario name to a list of line names that are
        missing at least one endpoint node;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_lines = {s: [] for s in scenarios_list}
    scenario_minor_lines = {s: [] for s in scenarios_list}

    for idx, item in lines_dict.items():
        numeric_fields = {
            "length": int,
            "capex": float,
            "transformer_capex": float,
            "fom": float,
            "vom": float,
            "lifetime": int,
            "discount_rate": float,
            "loss_factor": float,
            "initial_capacity": float,
            "max_build": float,
            "min_build": float,
        }

        fields_valid = True
        for field, cast in numeric_fields.items():
            try:
                val = cast(item[field])
                if "discount_rate" == field:
                    if not (0 <= val <= 1):
                        raise ValueError
                elif "loss_factor" == field:
                    if not (0 <= val < 1):
                        raise ValueError
                else:
                    if val < 0:
                        raise ValueError
            except ValueError:
                model_logger.error("'%s' must be a valid %s in appropriate range", field, cast.__name__)
                flag = False
                if field in ("min_build", "max_build"):
                    fields_valid = False

        if fields_valid:
            try:
                if float(item["min_build"]) > float(item["max_build"]):
                    model_logger.error("'min_build' must be less than or equal to 'max_build'")
                    flag = False
            except (TypeError, ValueError):
                pass

        for scenario in parse_list(item.get("scenarios")):
            if scenario in scenarios_list:
                scenario_lines[scenario].append(item["name"])

                for endpoint in ["node_start", "node_end"]:
                    node_val = item.get(endpoint)
                    if (node_val not in scenario_nodes[scenario]) and not is_nan(node_val):
                        model_logger.error(
                            "'%s' %s for line %s is not defined in scenario %s",
                            endpoint,
                            node_val,
                            item["name"],
                            scenario,
                        )
                        flag = False

                if any(is_nan(item.get(n)) for n in ["node_start", "node_end"]):
                    scenario_minor_lines[scenario].append(item["name"])
            else:
                model_logger.warning("'scenario' %s for line.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_lines, scenario_minor_lines, flag


def validate_generators(
    generators_dict: dict,
    scenarios_list: list,
    scenario_fuels: dict,
    scenario_lines: dict,
    scenario_nodes: dict,
    model_logger: Logger,
) -> tuple:
    """
    Validate all rows in generators.csv and build per-scenario generator indexes.

    Checks non-negative numeric fields, discount_rate range, valid unit_type, that
    min_build <= max_build, and that referenced node, fuel, and line exist within the
    relevant scenario.

    Parameters:
    -------
    generators_dict (dict): Mapping of row index to generator attribute dicts as loaded
        from generators.csv.
    scenarios_list (list): List of valid scenario names.
    scenario_fuels (dict): Mapping of scenario name to a list of valid fuel names.
    scenario_lines (dict): Mapping of scenario name to a list of valid line names.
    scenario_nodes (dict): Mapping of scenario name to a list of valid node names.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    tuple: A 3-tuple (scenario_generators, scenario_baseload, flag) where:
        scenario_generators (dict) maps scenario name to a list of generator names;
        scenario_baseload (dict) maps scenario name to a list of baseload generator names;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_generators = {s: [] for s in scenarios_list}
    scenario_baseload = {s: [] for s in scenarios_list}

    for idx, item in generators_dict.items():
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
                model_logger.error("'%s' must be float greater than or equal to 0", field)
                flag = False
                if field in ("min_build", "max_build"):
                    fields_valid = False

        if not validate_range(item["discount_rate"], 0, 1):
            model_logger.error("'discount_rate' must be float in range [0,1]")
            flag = False

        if not validate_enum(item["unit_type"], ["solar", "wind", "flexible", "baseload"]):
            model_logger.error("'unit_type' must be one of ['solar', 'wind', 'flexible', 'baseload']")
            flag = False

        if fields_valid:
            try:
                if float(item["min_build"]) > float(item["max_build"]):
                    model_logger.error("'min_build' must be less than or equal to 'max_build'")
                    flag = False
            except (TypeError, ValueError):
                pass

        for scenario in parse_list(item.get("scenarios")):
            if scenario in scenarios_list:
                if item["name"] in scenario_generators[scenario]:
                    model_logger.error("Duplicate generator name '%s' in scenario %s", item["name"], scenario)
                    flag = False
                else:
                    scenario_generators[scenario].append(item["name"])

                if item["unit_type"] == "baseload":
                    scenario_baseload[scenario].append(item["name"])

                if item["node"] not in scenario_nodes[scenario]:
                    model_logger.error(
                        "'node' %s for generator %s is not defined in scenario %s",
                        item["node"],
                        item["name"],
                        scenario,
                    )
                    flag = False

                if item["fuel"] not in scenario_fuels[scenario]:
                    model_logger.error(
                        "'fuel' %s for generator %s is not defined in scenario %s",
                        item["fuel"],
                        item["name"],
                        scenario,
                    )
                    flag = False

                if item["line"] not in scenario_lines[scenario]:
                    model_logger.error(
                        "'line' %s for generator %s is not defined in scenario %s",
                        item["line"],
                        item["name"],
                        scenario,
                    )
                    flag = False
            else:
                model_logger.warning("'scenario' %s for generator.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_generators, scenario_baseload, flag


def validate_storages(
    storages_dict: dict, scenarios_list: list, scenario_nodes: dict, scenario_lines: dict, model_logger: Logger
) -> tuple:
    """
    Validate all rows in storages.csv and build a per-scenario storage index.

    Checks non-negative capacity and cost fields, that min/max build pairs are ordered
    correctly, that lifetime and duration are non-negative integers, that efficiency
    values are in [0, 1], and that referenced node and line exist within the relevant
    scenario.

    Parameters:
    -------
    storages_dict (dict): Mapping of row index to storage attribute dicts as loaded
        from storages.csv.
    scenarios_list (list): List of valid scenario names.
    scenario_nodes (dict): Mapping of scenario name to a list of valid node names.
    scenario_lines (dict): Mapping of scenario name to a list of valid line names.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    tuple: A 2-tuple (scenario_storages, flag) where:
        scenario_storages (dict) maps scenario name to a list of storage names;
        flag (bool) is False if any validation error was found, True otherwise.
    """
    flag = True
    scenario_storages = {s: [] for s in scenarios_list}

    for idx, item in storages_dict.items():
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
                model_logger.error("'%s' must be float >= 0", field)
                flag = False
                if field in ("min_build_p", "max_build_p", "min_build_e", "max_build_e"):
                    build_fields_valid = False

        if build_fields_valid:
            for min_field, max_field in [("min_build_p", "max_build_p"), ("min_build_e", "max_build_e")]:
                try:
                    if float(item[min_field]) > float(item[max_field]):
                        model_logger.error("'%s' must be <= '%s'", min_field, max_field)
                        flag = False
                except (TypeError, ValueError):
                    pass

        for field in ["lifetime", "duration"]:
            try:
                if int(item[field]) < 0:
                    model_logger.error("'%s' must be int >= 0", field)
                    flag = False
            except (TypeError, ValueError):
                model_logger.error("'%s' must be a valid integer", field)
                flag = False

        for efficiency in ["charge_efficiency", "discharge_efficiency"]:
            if not validate_range(item[efficiency], 0, 1):
                model_logger.error("'%s' must be float in [0,1]", efficiency)
                flag = False

        if not validate_range(item["discount_rate"], 0, 1):
            model_logger.error("'discount_rate' must be float in [0,1]")
            flag = False

        for scenario in parse_list(item.get("scenarios")):
            if scenario in scenarios_list:
                if item["name"] in scenario_storages[scenario]:
                    model_logger.error("Duplicate storage name '%s' in scenario %s", item["name"], scenario)
                    flag = False
                else:
                    scenario_storages[scenario].append(item["name"])

                if item["node"] not in scenario_nodes[scenario]:
                    model_logger.error(
                        "'node' %s for storage %s is not defined in scenario %s",
                        item["node"],
                        item["name"],
                        scenario,
                    )
                    flag = False

                if item["line"] not in scenario_lines[scenario]:
                    model_logger.error(
                        "'line' %s for storage %s is not defined in scenario %s",
                        item["line"],
                        item["name"],
                        scenario,
                    )
                    flag = False
            else:
                model_logger.warning("'scenario' %s for storage.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_storages, flag


def validate_initial_guess(
    x0s_dict: dict,
    scenarios_list: list,
    scenario_generators: dict,
    scenario_storages: dict,
    scenario_lines: dict,
    scenario_baseload: dict,
    scenario_minor_lines: dict,
    model_logger: Logger,
) -> bool:
    """
    Validate all rows in initial_guess.csv.

    Checks that each x_0 vector has the correct length for its scenario (one value per
    generator + two per storage + one per major line), and that every scenario
    defined in scenarios.csv has a corresponding entry.

    Parameters:
    -------
    x0s_dict (dict): Mapping of row index to initial-guess attribute dicts as loaded
        from initial_guess.csv.
    scenarios_list (list): List of valid scenario names.
    scenario_generators (dict): Mapping of scenario name to a list of generator names.
    scenario_storages (dict): Mapping of scenario name to a list of storage names.
    scenario_lines (dict): Mapping of scenario name to a list of line names.
    scenario_baseload (dict): Mapping of scenario name to a list of baseload generator names.
    scenario_minor_lines (dict): Mapping of scenario name to a list of minor line names
        (lines without both endpoints).
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    bool: True if all initial-guess entries are valid and complete, False otherwise.
    """
    flag = True
    initial_guess_scenarios = []

    for item in x0s_dict.values():
        scenario = item["scenario"]

        if scenario not in scenarios_list:
            model_logger.warning("'scenario' %s in initial_guess.csv not defined in scenarios.csv", scenario)

        initial_guess_scenarios.append(scenario)

        x0 = parse_list(item["x_0"])

        bound_length = len(
            scenario_generators[scenario]
            + scenario_storages[scenario]
            + scenario_storages[scenario]
            + scenario_lines[scenario]
        ) - len(scenario_minor_lines[scenario])

        if x0 and len(x0) != bound_length:
            model_logger.error(
                "Initial guess 'x_0' for scenario %s contains %d elements, expected %d",
                scenario,
                len(x0),
                bound_length,
            )
            flag = False

    for scenario in scenarios_list:
        if scenario not in initial_guess_scenarios:
            model_logger.error("'scenario' %s is defined in scenarios.csv but missing from initial_guess.csv", scenario)
            flag = False

    return flag


def validate_datafiles_config(
    scenario_filenames: list, scenario_datafile_types: list, model_logger: Logger, datafiles_directory: str
) -> bool:
    """
    Check that all referenced data files exist on disk and have valid types.

    Parameters:
    -------
    scenario_filenames (list): List of filenames expected to be present in
        datafiles_directory.
    scenario_datafile_types (list): List of datafile type strings corresponding to each
        filename. Valid values are "demand", "generation", and "flexible_annual_limit".
    model_logger (logging.Logger): Logger instance for recording errors and warnings.
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
            model_logger.error("Missing file data/%s", fn)
            flag = False

    for dtype in scenario_datafile_types:
        if not dtype or dtype not in valid_types:
            model_logger.error("Invalid or missing datafile_type '%s'", dtype)
            flag = False

    return flag


def validate_electricity(file_path: str, node_list: list, model_logger: Logger) -> bool:
    """
    Validate a demand profile CSV file.

    Checks that the required time columns are present and contain valid values, and that each
    expected node has a column.

    Parameters:
    -------
    file_path (str): Absolute path to the demand CSV file.
    node_list (list): List of node names expected as columns in the file. If None, column
        name validation is skipped and only value ranges are checked.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    bool: True if the file is structurally valid and all demand values are non-negative,
        False otherwise.
    """
    flag = True

    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    except Exception as e:
        model_logger.error("Could not read demand file '%s': %s", file_path, e)
        return False

    required_time_cols = ["Year", "Month", "Day", "Interval"]
    for col in required_time_cols:
        if col not in df.columns:
            model_logger.error("Demand file '%s' is missing required column '%s'", file_path, col)
            flag = False

    if not flag:
        return False

    try:
        if not (df["Year"] >= 1).all():
            model_logger.error("'Year' values in '%s' must be positive integers", file_path)
            flag = False
    except Exception:
        model_logger.error("'Year' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not df["Month"].isin(range(1, 13)).all():
            model_logger.error("'Month' values in '%s' must be in range [1, 12]", file_path)
            flag = False
    except Exception:
        model_logger.error("'Month' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not df["Day"].between(1, 31).all():
            model_logger.error("'Day' values in '%s' must be in range [1, 31]", file_path)
            flag = False
    except Exception:
        model_logger.error("'Day' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not (df["Interval"] >= 1).all():
            model_logger.error("'Interval' values in '%s' must be >= 1", file_path)
            flag = False
    except Exception:
        model_logger.error("'Interval' column in '%s' contains non-numeric values", file_path)
        flag = False

    if node_list is not None:
        for node in node_list:
            if node not in df.columns:
                model_logger.error("Demand file '%s' is missing column for node '%s'", file_path, node)
                flag = False

    data_cols = [c for c in df.columns if c not in required_time_cols]
    for col in data_cols:
        if df[col].isna().any():
            model_logger.error("Column '%s' in demand file '%s' contains NaN values", col, file_path)
            flag = False
        elif not pd.api.types.is_numeric_dtype(df[col]):
            model_logger.error("Column '%s' in demand file '%s' contains non-numeric values", col, file_path)
            flag = False

    return flag


def validate_generation(
    file_path: str, solar_list: list, wind_list: list, baseload_list: list, model_logger: Logger
) -> bool:
    """
    Validate a generation trace CSV file for solar, wind, or baseload units.

    Checks that the required time columns are present and contain valid values, that all
    data columns contain capacity factors in [0, 1] with no missing data, and warns if
    any data column does not correspond to a configured solar, wind, or baseload generator.

    Parameters:
    -------
    file_path (str): Absolute path to the generation trace CSV file.
    solar_list (list): List of solar generator names for the scenario. If all three
        generator lists are None, column name validation is skipped.
    wind_list (list): List of wind generator names for the scenario.
    baseload_list (list): List of baseload generator names for the scenario.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

    Returns:
    -------
    bool: True if the file is structurally valid and all capacity factors are in [0, 1],
        False otherwise.
    """
    flag = True

    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    except Exception as e:
        model_logger.error("Could not read generation file '%s': %s", file_path, e)
        return False

    required_time_cols = ["Year", "Month", "Day", "Interval"]
    for col in required_time_cols:
        if col not in df.columns:
            model_logger.error("Generation file '%s' is missing required column '%s'", file_path, col)
            flag = False

    if not flag:
        return False

    try:
        if not (df["Year"] >= 1).all():
            model_logger.error("'Year' values in '%s' must be positive integers", file_path)
            flag = False
    except Exception:
        model_logger.error("'Year' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not df["Month"].isin(range(1, 13)).all():
            model_logger.error("'Month' values in '%s' must be in range [1, 12]", file_path)
            flag = False
    except Exception:
        model_logger.error("'Month' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not df["Day"].between(1, 31).all():
            model_logger.error("'Day' values in '%s' must be in range [1, 31]", file_path)
            flag = False
    except Exception:
        model_logger.error("'Day' column in '%s' contains non-numeric values", file_path)
        flag = False

    try:
        if not (df["Interval"] >= 1).all():
            model_logger.error("'Interval' values in '%s' must be >= 1", file_path)
            flag = False
    except Exception:
        model_logger.error("'Interval' column in '%s' contains non-numeric values", file_path)
        flag = False

    data_cols = [c for c in df.columns if c not in required_time_cols]

    if solar_list is not None and wind_list is not None and baseload_list is not None:
        known_generators = set(solar_list) | set(wind_list) | set(baseload_list)
        for col in data_cols:
            if col not in known_generators:
                model_logger.warning(
                    "Column '%s' in generation file '%s' does not match any configured generator",
                    col,
                    file_path,
                )

    for col in data_cols:
        if df[col].isna().any():
            model_logger.error("Column '%s' in generation file '%s' contains NaN values", col, file_path)
            flag = False
        elif not df[col].between(0, 1).all():
            model_logger.error(
                "Column '%s' in generation file '%s' contains values outside [0, 1]",
                col,
                file_path,
            )
            flag = False

    return flag


def validate_flexible_limits(file_path: str, flexible_list: list, model_logger: Logger) -> bool:
    """
    Validate a flexible annual generation limit CSV file.

    Checks that a Year column is present and contains valid values, that each expected
    flexible generator has a column, and that all limit values are non-negative with no
    missing data. Unlike demand and generation files, this file has one row per year
    rather than per time interval.

    Parameters:
    -------
    file_path (str): Absolute path to the flexible annual limit CSV file.
    flexible_list (list): List of flexible generator names expected as columns in the file.
        If None, column name validation is skipped and only value ranges are checked.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.

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
        model_logger.error("Could not read flexible limits file '%s': %s", file_path, e)
        return False

    if "Year" not in df.columns:
        model_logger.error("Flexible limits file '%s' is missing required column 'Year'", file_path)
        return False

    try:
        if not (df["Year"] >= 1).all():
            model_logger.error("'Year' values in '%s' must be positive integers", file_path)
            flag = False
    except Exception:
        model_logger.error("'Year' column in '%s' contains non-numeric values", file_path)
        flag = False

    if flexible_list is not None:
        for gen in flexible_list:
            if gen not in df.columns:
                model_logger.error(
                    "Flexible limits file '%s' is missing column for generator '%s'",
                    file_path,
                    gen,
                )
                flag = False

    data_cols = [c for c in df.columns if c != "Year"]
    for col in data_cols:
        if df[col].isna().any():
            model_logger.error("Column '%s' in flexible limits file '%s' contains NaN values", col, file_path)
            flag = False
        elif (df[col] < 0).any():
            model_logger.error("Column '%s' in flexible limits file '%s' contains negative values", col, file_path)
            flag = False

    return flag


def validate_config(model_data: ModelData) -> bool:
    """
    Run full validation of all configuration CSV tables in a ModelData instance.

    Validates config, scenarios, fuels, lines, generators, storages, and initial_guess
    in dependency order, propagating cross-table references (e.g. scenario node lists)
    between validation steps.

    Parameters:
    -------
    model_data (ModelData): A fully loaded ModelData instance.

    Returns:
    -------
    bool: True if all configuration tables pass validation, False if any errors were found.
    """
    config_flag = True
    model_logger = model_data.logger

    if not validate_model_config(model_data.config, model_logger):
        model_logger.error("config.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("config.csv validated!")

    scenarios_list, scenario_nodes, _, flag = validate_scenarios(model_data.scenarios, model_logger)
    if not flag:
        model_logger.error("scenarios.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("scenarios.csv validated!")

    scenario_fuels, flag = validate_fuels(model_data.fuels, scenarios_list, model_logger)
    if not flag:
        model_logger.error("fuels.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("fuels.csv validated!")

    scenario_lines, scenario_minor_lines, flag = validate_lines(
        model_data.lines, scenarios_list, scenario_nodes, model_logger
    )
    if not flag:
        model_logger.error("lines.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("lines.csv validated!")

    scenario_generators, scenario_baseload, flag = validate_generators(
        model_data.generators, scenarios_list, scenario_fuels, scenario_lines, scenario_nodes, model_logger
    )
    if not flag:
        model_logger.error("generators.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("generators.csv validated!")

    scenario_storages, flag = validate_storages(
        model_data.storages, scenarios_list, scenario_nodes, scenario_lines, model_logger
    )
    if not flag:
        model_logger.error("storages.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("storages.csv validated!")

    if not validate_initial_guess(
        model_data.x0s,
        scenarios_list,
        scenario_generators,
        scenario_storages,
        scenario_lines,
        scenario_baseload,
        scenario_minor_lines,
        model_logger,
    ):
        model_logger.error("initial_guess.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("initial_guess.csv validated!")

    return config_flag


def validate_data(
    all_datafiles: dict,
    scenario_name: str,
    model_logger: Logger,
    datafiles_directory: str,
    node_list: list = None,
    solar_list: list = None,
    wind_list: list = None,
    baseload_list: list = None,
    flexible_list: list = None,
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
    all_datafiles (dict): Mapping of row index to datafile attribute dicts as loaded
        from datafiles.csv.
    scenario_name (str): The name of the scenario whose data files should be validated.
    model_logger (logging.Logger): Logger instance for recording errors and warnings.
    datafiles_directory (str): Absolute path to the directory containing data files.
    node_list (list): List of node names expected in demand files for this scenario.
        Defaults to None (skips column name validation).
    solar_list (list): List of solar generator names for this scenario.
        Defaults to None (skips column name validation).
    wind_list (list): List of wind generator names for this scenario.
        Defaults to None (skips column name validation).
    baseload_list (list): List of baseload generator names for this scenario.
        Defaults to None (skips column name validation).
    flexible_list (list): List of flexible generator names expected in annual limit files.
        Defaults to None (skips column name validation).

    Returns:
    -------
    bool: True if all data files for the scenario are valid, False otherwise.
    """
    flag = True
    scenario_filenames = []
    scenario_datafile_types = []
    scenario_files_by_type = {"demand": [], "generation": [], "flexible_annual_limit": []}

    for item in all_datafiles.values():
        scenario_list = parse_comma_separated(item["scenarios"])
        if scenario_name in scenario_list:
            filename = item["filename"]
            dtype = item["datafile_type"]
            scenario_filenames.append(filename)
            scenario_datafile_types.append(dtype)
            if dtype in scenario_files_by_type:
                scenario_files_by_type[dtype].append(filename)

    if not validate_datafiles_config(scenario_filenames, scenario_datafile_types, model_logger, datafiles_directory):
        model_logger.error("datafiles.csv contains errors for scenario %s.", scenario_name)
        flag = False
    else:
        model_logger.info("datafiles.csv validated for scenario %s!", scenario_name)

    for filename in scenario_files_by_type["demand"]:
        file_path = os.path.join(datafiles_directory, filename)
        if not validate_electricity(file_path, node_list, model_logger):
            model_logger.error("Demand file '%s' contains errors for scenario %s.", filename, scenario_name)
            flag = False
        else:
            model_logger.info("Demand file '%s' validated for scenario %s!", filename, scenario_name)

    for filename in scenario_files_by_type["generation"]:
        file_path = os.path.join(datafiles_directory, filename)
        if not validate_generation(file_path, solar_list, wind_list, baseload_list, model_logger):
            model_logger.error("Generation file '%s' contains errors for scenario %s.", filename, scenario_name)
            flag = False
        else:
            model_logger.info("Generation file '%s' validated for scenario %s!", filename, scenario_name)

    for filename in scenario_files_by_type["flexible_annual_limit"]:
        file_path = os.path.join(datafiles_directory, filename)
        if not validate_flexible_limits(file_path, flexible_list, model_logger):
            model_logger.error("Flexible limits file '%s' contains errors for scenario %s.", filename, scenario_name)
            flag = False
        else:
            model_logger.info("Flexible limits file '%s' validated for scenario %s!", filename, scenario_name)

    return flag
