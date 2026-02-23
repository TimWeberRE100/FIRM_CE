import csv
import os
from typing import Dict, List

import numpy as np

from firm_ce.common.constants import PENALTY_MULTIPLIER
from firm_ce.common.typing import (
    BandCandidates_Type,
    BroadOptimumVars_Type,
    EvaluationRecord_Type,
)
from firm_ce.optimisation.single_time import Solution, parallel_wrapper
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType


def near_optimum_path(root: str, scenario_name: str):
    """
    Construct and create the output directory path for near-optimum results.

    Parameters:
    -------
    root (str): Root subdirectory name under the results directory.
    scenario_name (str): Name of the scenario, used as a further subdirectory.

    Returns:
    -------
    str: The path to the created directory.

    Side-effects:
    -------
    Creates the directory at the returned path if it does not already exist.

    Exceptions:
    -------
    None.
    """
    base = os.path.join("results", root, scenario_name)
    os.makedirs(base, exist_ok=True)
    return base


def create_broad_optimum_vars_record(
    candidate_x_idx: int, asset_name: str, near_optimum_check: bool, near_optimum_group: str
) -> BroadOptimumVars_Type:
    """
    Create a record for a single broad optimum decision variable.

    Parameters:
    -------
    candidate_x_idx (int): Index of the decision variable in the candidate solution vector.
    asset_name (str): Name of the asset associated with this decision variable.
    near_optimum_check (bool): Whether this variable participates in near-optimum exploration.
    near_optimum_group (str): Group key used to aggregate variables during near-optimum analysis.

    Returns:
    -------
    BroadOptimumVars_Type: A tuple of (candidate_x_idx, asset_name, near_optimum_check, near_optimum_group).

    Side-effects:
    -------
    None.

    Exceptions:
    -------
    None.
    """
    return (
        candidate_x_idx,
        asset_name,
        near_optimum_check,
        near_optimum_group,
    )


def build_broad_optimum_var_info(
    fleet: Fleet_InstanceType, network: Network_InstanceType
) -> List[BroadOptimumVars_Type]:
    """
    Build a list of records mapping each decision variable index to its metadata.

    Iterates over all generators, storage power capacity variables, storage energy capacity variables,
    and major transmission lines in the system to produce one record per decision
    variable, capturing its name, near-optimum participation flag, and group key.

    Parameters:
    -------
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    network (Network_InstanceType): An instance of the Network jitclass.

    Returns:
    -------
    List[BroadOptimumVars_Type]: Ordered list of records, one per decision variable,
        each containing (candidate_x_idx, asset_name, near_optimum_check, group).

    Side-effects:
    -------
    None.

    Exceptions:
    -------
    None.
    """

    broad_optimum_var_info = []

    for generator in fleet.generators.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                generator.candidate_x_idx, generator.name, generator.near_optimum_check, generator.group
            )
        )

    for storage in fleet.storages.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_p_x_idx, storage.name, storage.near_optimum_check, storage.group
            )
        )

    for storage in fleet.storages.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_e_x_idx, storage.name, storage.near_optimum_check, storage.group
            )
        )

    for line in network.major_lines.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(line.candidate_x_idx, line.name, line.near_optimum_check, line.group)
        )
    return broad_optimum_var_info


def create_evaluation_record(
    group_key: str,
    band_type: str,
    population_lcoes: List[float],
    de_population_penalties: List[float],
    band_population_penalties: List[float],
    band_population_candidates: List[List[float]],
    solution_index: int,
    target_group_var_sum: float | None = None,
) -> EvaluationRecord_Type:
    """
    Create a single evaluation record for a feasible candidate solution.

    Parameters:
    -------
    group_key (str): The group being explored (e.g. asset group name).
    band_type (str): Type of band search, one of "min", "max", or "Midpoint N".
    population_lcoes (List[float]): LCOE values for each candidate in the population.
    de_population_penalties (List[float]): Operational penalty values from differential
        evolution for each candidate in the population.
    band_population_penalties (List[float]): Band constraint penalty values for each
        candidate in the population.
    band_population_candidates (List[List[float]]): 2-D array of candidate solution
        vectors (rows = variables, columns = candidates).
    solution_index (int): Index into the population arrays for the candidate to record.
    target_group_var_sum (float | None): Target sum of group variables for midpoint
        searches. Stored as "N/A" when None.

    Returns:
    -------
    EvaluationRecord_Type: A tuple of (group_key, band_type, target_group_var_sum,
        lcoe, de_penalty, band_penalty, candidate_x_array).

    Side-effects:
    -------
    None.

    Exceptions:
    -------
    None.
    """
    return (
        group_key,
        band_type,
        target_group_var_sum if target_group_var_sum else "N/A",
        float(population_lcoes[solution_index]),
        float(de_population_penalties[solution_index]),
        float(band_population_penalties[solution_index]),
        band_population_candidates[:, solution_index].copy(),
    )


def broad_optimum_objective(
    band_population_candidates: List[List[float]],  # 2-D array to allow vectorized DE
    differential_evolution_args,
    group_key: str,
    band_lcoe_max: float,
    bo_group_orders: List[int],
    evaluation_records: List[EvaluationRecord_Type],
    band_type: str,
    target_group_var_sum: float | None = None,
    midpoint_number: int | None = None,
) -> float:
    """
    Objective function for broad optimum band and midpoint searches.

    Evaluates a population of candidate solutions, appends feasible solutions to
    evaluation_records, and returns a scalar fitness array guiding differential
    evolution. For "min" and "max" band searches the fitness steers towards the
    minimum or maximum group variable sum respectively. For "midpoint" searches
    the fitness steers towards a specified target group variable sum.

    Parameters:
    -------
    band_population_candidates (List[List[float]]): 2-D array of candidate solution
        vectors (rows = variables, columns = candidates).
    differential_evolution_args: Arguments forwarded to parallel_wrapper for solution
        evaluation (fleet, network, etc.).
    group_key (str): The asset group currently being explored.
    band_lcoe_max (float): Maximum allowable LCOE for a feasible solution. Candidates
        exceeding this incur a band penalty.
    bo_group_orders (List[int]): Row indices into band_population_candidates
        corresponding to the variables in the current group.
    evaluation_records (List[EvaluationRecord_Type]): Accumulator list; feasible
        candidate records are appended in-place.
    band_type (str): One of "min", "max", or "midpoint".
    target_group_var_sum (float | None): Target sum of group variables used when
        band_type is "midpoint". Ignored otherwise.
    midpoint_number (int | None): Label index for the midpoint, used to name the
        band_type field in appended records. Ignored when band_type is not "midpoint".

    Returns:
    -------
    float: Array of population energies, one per candidate, to minimise during differential
        evolution. Returns None if band_type is unrecognised.

    Side-effects:
    -------
    Feasible candidates (zero operational and band penalties) are appended to
    evaluation_records.

    Exceptions:
    -------
    None.
    """
    _, population_lcoes, de_population_penalties = parallel_wrapper(
        band_population_candidates, *differential_evolution_args
    )
    group_var_sums = band_population_candidates[bo_group_orders, :].sum(axis=0)
    band_population_penalties = np.maximum(0, population_lcoes - band_lcoe_max) * PENALTY_MULTIPLIER

    match band_type:
        case "min" | "max":
            for candidate_x in range(band_population_candidates.shape[1]):
                if not (
                    de_population_penalties[candidate_x] <= 0.001 and band_population_penalties[candidate_x] <= 0.001
                ):
                    continue
                evaluation_records.append(
                    create_evaluation_record(
                        group_key,
                        band_type,
                        population_lcoes,
                        de_population_penalties,
                        band_population_penalties,
                        band_population_candidates,
                        candidate_x,
                    )
                )
        case "midpoint":
            target_penalties = np.abs(group_var_sums - target_group_var_sum)
            for candidate_x in range(band_population_candidates.shape[1]):
                if not (
                    de_population_penalties[candidate_x] <= 0.001 and band_population_penalties[candidate_x] <= 0.001
                ):
                    continue
                evaluation_records.append(
                    create_evaluation_record(
                        group_key,
                        f"Midpoint {midpoint_number}",
                        population_lcoes,
                        de_population_penalties,
                        band_population_penalties,
                        band_population_candidates,
                        candidate_x,
                        target_group_var_sum,
                    )
                )

    match band_type:
        case "min":
            return band_population_penalties + de_population_penalties + group_var_sums
        case "max":
            return band_population_penalties + de_population_penalties - group_var_sums
        case "midpoint":
            return band_population_penalties + de_population_penalties + target_penalties
        case _:
            return None


def write_broad_optimum_records(
    scenario_name: str,
    evaluation_records: List[EvaluationRecord_Type],
    broad_optimum_var_info: List[BroadOptimumVars_Type],
) -> None:
    """
    Write all broad optimum evaluation records to a CSV file.

    Parameters:
    -------
    scenario_name (str): Name of the scenario, used to determine the output directory.
    evaluation_records (List[EvaluationRecord_Type]): List of feasible candidate
        records produced during the broad optimum search.
    broad_optimum_var_info (List[BroadOptimumVars_Type]): Variable metadata list used
        to generate column headers (one column per asset variable).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Writes the file near_optimal_space.csv inside the near_optimum
    results directory for the given scenario.

    Exceptions:
    -------
    None.
    """
    space_dir = near_optimum_path("near_optimum", scenario_name)

    space_path = os.path.join(space_dir, "near_optimal_space.csv")
    with open(space_path, "w", newline="") as f_space:
        writer_space = csv.writer(f_space)
        writer_space.writerow(
            [
                "Group",
                "Band_Type",
                "LCOE [$/MWh]",
                "Operational_Penalty",
                "Band_Penalty",
                *[f"{asset_name}" for _, asset_name, _, _ in broad_optimum_var_info],
            ]
        )
        for group, band_type, _, lcoe, de_penalty, band_penalty, candidate_x in evaluation_records:
            writer_space.writerow([group, band_type, lcoe, de_penalty, band_penalty, *candidate_x])
    return None


def get_broad_optimum_bands_path(scenario_name: str) -> str:
    """
    Return the file path for the near-optimal bands CSV file.

    Parameters:
    -------
    scenario_name (str): Name of the scenario.

    Returns:
    -------
    str: Absolute or relative path to near_optimal_bands.csv in the scenario's
        near_optimum results directory.

    Side-effects:
    -------
    Creates the near_optimum results directory for the scenario if it does not
    already exist.

    Exceptions:
    -------
    None.
    """
    space_dir = near_optimum_path("near_optimum", scenario_name)
    return os.path.join(space_dir, "near_optimal_bands.csv")


def write_broad_optimum_bands(
    scenario_name: str,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
    bands: BandCandidates_Type,
    de_args,
    band_lcoe_max: float,
    groups: Dict[str, List[int]],
) -> None:
    """
    Evaluate and write the min/max band candidate solutions to a CSV file.

    For each group, re-evaluates the stored min and max candidate solutions to
    compute their LCOE and penalty values, then writes one row per band endpoint
    to the near-optimal bands CSV.

    Parameters:
    -------
    scenario_name (str): Name of the scenario, used to determine the output path.
    broad_optimum_var_info (List[BroadOptimumVars_Type]): Variable metadata list
        used to map variable indices to asset names and filter near-optimum variables.
    bands (BandCandidates_Type): Dictionary mapping group keys to a tuple of
        (min_candidate_x, max_candidate_x) solution vectors.
    de_args: Arguments forwarded to Solution for evaluation (fleet, network, etc.).
    band_lcoe_max (float): Maximum allowable LCOE used to compute the band penalty.
    groups (Dict[str, List[int]]): Mapping from group key to list of decision
        variable indices belonging to that group.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Writes the file near_optimal_bands.csv inside the near_optimum
    results directory for the given scenario.

    Exceptions:
    -------
    None.
    """
    bands_path = get_broad_optimum_bands_path(scenario_name)
    near_optimal_asset_names = [
        asset_name for _, asset_name, near_optimum_check, _ in broad_optimum_var_info if near_optimum_check
    ]
    names_to_columns = {name: col for col, name in enumerate(near_optimal_asset_names)}

    with open(bands_path, "w", newline="") as f_bands:
        writer_bands = csv.writer(f_bands)
        header = [
            "Group",
            "Band_Type",
            "LCOE [$/MWh]",
            "Operational_Penalty",
            "Band_Penalty",
        ] + near_optimal_asset_names
        writer_bands.writerow(header)

        for group, (candidate_x_min, candidate_x_max) in bands.items():
            for band_type, candidate_x in (("min", candidate_x_min), ("max", candidate_x_max)):
                solution = Solution(candidate_x, *de_args)
                solution.evaluate()
                band_penalty = max(0, solution.lcoe - band_lcoe_max) * PENALTY_MULTIPLIER
                row = [group, band_type, solution.lcoe, solution.penalties, band_penalty]
                vals = [""] * len(near_optimal_asset_names)
                for candidate_x_idx in groups[group]:
                    _, asset_name, _, _ = broad_optimum_var_info[candidate_x_idx]
                    col = names_to_columns[asset_name]
                    vals[col] = candidate_x[candidate_x_idx]
                writer_bands.writerow(row + vals)
    return None


def read_broad_optimum_bands(
    scenario_name: str,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
) -> Dict[str, Dict[str, float]]:
    """
    Read the near-optimal bands CSV and return the min/max group variable sums.

    Parameters:
    -------
    scenario_name (str): Name of the scenario used to locate the bands CSV.
    broad_optimum_var_info (List[BroadOptimumVars_Type]): Variable metadata list
        used to identify which assets belong to each group.

    Returns:
    -------
    Dict[str, Dict[str, float]]: Dictionary mapping each group key to a nested dict with keys "min" and
        "max", each holding the summed decision variable values for that band
        endpoint across all near-optimum assets in the group.

    Side-effects:
    -------
    None.

    Exceptions:
    -------
    None.
    """
    bands_path = get_broad_optimum_bands_path(scenario_name)

    group_names = {}
    group_bands = {}
    for _, name, near_optimum_check, group in broad_optimum_var_info:
        if not near_optimum_check:
            continue
        group_names.setdefault(group, []).append(name)

    with open(bands_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["Group"]
            band_type = row["Band_Type"]
            values = [float(row[variable_name]) for variable_name in group_names.get(group, [])]
            s = sum(values)
            group_bands.setdefault(group, {"min": None, "max": None})[band_type] = s
    return group_bands


def get_midpoint_csv_path(scenario_name: str) -> str:
    """
    Return the file path for the midpoint exploration CSV file.

    Parameters:
    -------
    scenario_name (str): Name of the scenario.

    Returns:
    -------
    str: Absolute or relative path to midpoint_space.csv in the scenario's
        midpoint_explore results directory.

    Side-effects:
    -------
    Creates the midpoint_explore results directory for the scenario if it does
    not already exist.

    Exceptions:
    -------
    None.
    """
    midpoint_dir = near_optimum_path("midpoint_explore", scenario_name)
    return os.path.join(midpoint_dir, "midpoint_space.csv")


def create_midpoint_csv(scenario_name: str, broad_optimum_var_info: List[BroadOptimumVars_Type]) -> str:
    """
    Create an empty midpoint exploration CSV file with the appropriate header row.

    Parameters:
    -------
    scenario_name (str): Name of the scenario used to determine the output path.
    broad_optimum_var_info (List[BroadOptimumVars_Type]): Variable metadata list
        used to generate one column per asset variable in the header.

    Returns:
    -------
    str: Path to the created CSV file.

    Side-effects:
    -------
    Creates the file midpoint_space.csv inside the midpoint_explore results directory for the given scenario.

    Exceptions:
    -------
    None.
    """
    csv_path = get_midpoint_csv_path(scenario_name)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Group",
                "Midpoint ID",
                "Target Value [GW or GWh]",
                "LCOE [$/MWh]",
                "Operational_Penalty",
                "Band_Penalty",
                *[f"{asset_name}" for _, asset_name, _, _ in broad_optimum_var_info],
            ]
        )
    return csv_path


def create_groups_dict(broad_optimum_var_info: List[BroadOptimumVars_Type]) -> Dict[str, List[int]]:
    """
    Build a mapping from group keys to lists of decision variable indices.

    Only variables with near_optimum_check enabled are included. Variables with
    no group key are placed in their own singleton group keyed by their variable
    index.

    Parameters:
    -------
    broad_optimum_var_info (List[BroadOptimumVars_Type]): Variable metadata list
        containing (candidate_x_idx, asset_name, near_optimum_check, group) tuples.

    Returns:
    -------
    Dict[str, List[int]]: Mapping from group key (str or int) to a list of decision variable
        indices belonging to that group.

    Side-effects:
    -------
    None.

    Exceptions:
    -------
    None.
    """
    groups = {}
    for record in broad_optimum_var_info:
        candidate_x_idx, _, near_optimum_check, group = record

        if not near_optimum_check:
            continue
        key = group or candidate_x_idx
        groups.setdefault(key, []).append(candidate_x_idx)
    return groups


def append_to_midpoint_csv(
    scenario_name: str,
    evaluation_records: List[EvaluationRecord_Type],
) -> None:
    """
    Append midpoint evaluation records to the midpoint exploration CSV file.

    Parameters:
    -------
    scenario_name (str): Name of the scenario used to locate the CSV file.
    evaluation_records (List[EvaluationRecord_Type]): List of feasible candidate
        records to append, each containing group, band_type, target_value, lcoe,
        de_penalty, band_penalty, and the candidate solution vector.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Appends rows to midpoint_space.csv in the midpoint_explore results directory
    for the given scenario.

    Exceptions:
    -------
    None.
    """
    csv_path = get_midpoint_csv_path(scenario_name)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        for group, band_type, target_value, lcoe, de_penalty, band_penalty, candidate_x in evaluation_records:
            writer.writerow([group, band_type, target_value, lcoe, de_penalty, band_penalty, *candidate_x])
    return None
