import math
import os
import shutil
import time
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from sklearn.cluster import MiniBatchKMeans

from firm_ce.common.logging import get_logger
from firm_ce.fast_methods import static_m
from firm_ce.optimisation.capacity_expansion import (
    _active_vintages,  # noqa: F401 (imported for re-use within this module)
    _append_pathway_records,
    _build_effective_year_bounds,
    _initialise_pathway_records,
    _prepare_existing_capacity_arrays,
    _update_vintages,
    _write_pathway_records,
    evaluate_capacity_expansion_solution,
)
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.interventions import ScenarioScenarioInterventions
from firm_ce.system.parameters import ModelConfig, ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_InterventionEntry = namedtuple(
    "_InterventionEntry",
    [
        "candidate_x_idx",  # int: index into the candidate-x vector
        "unit_size",  # float: GW for generator/storage-power, GWh for storage-energy
        "max_repetitions",  # int: floor(max_build / unit_size) — max uses per set
        "asset_type",  # str: 'generator', 'storage_power', or 'storage_energy'
        "asset_name",  # str: human-readable name for logging
        "asset_order",  # int: order key in fleet dict
    ],
)

# Typed alias used by the vintage helpers imported from capacity_expansion
_VintageList = List[List[Tuple[int, float, int]]]


# ---------------------------------------------------------------------------
# Intervention generation
# ---------------------------------------------------------------------------


def _generate_interventions_for_year(
    year_idx: int,
    fleet_static: Fleet_InstanceType,
    system_interventions: Optional[ScenarioScenarioInterventions],
) -> List[_InterventionEntry]:
    """
    Build the list of available discrete interventions for a given year.

    Each entry represents the addition of one discrete build step of a specific
    asset. Assets and their step sizes are taken from the ScenarioInterventions object
    built from interventions.csv. An asset is included only when at least one
    step can be built within its max_build limit for the year.

    Parameters:
    -------
    year_idx (int): Year index (0-based offset from first_year).
    fleet_static (Fleet_InstanceType): Static fleet data, used to read max_build
        limits and candidate vector indices for each asset.
    system_interventions (ScenarioInterventions | None): ScenarioInterventions container for the
        scenario. Returns an empty list when None.

    Returns:
    -------
    List[_InterventionEntry]: List of available intervention entries for the year.
    """
    if system_interventions is None:
        return []

    entries: List[_InterventionEntry] = []

    for iv in system_interventions.interventions.get(year_idx, {}).values():
        for order, cap in iv.generator_capacities.items():
            generator = fleet_static.generators[order]
            mb = float(generator.max_build[year_idx])
            if cap > 0:
                max_reps = int(math.floor(mb / cap))
                if max_reps > 0:
                    entries.append(
                        _InterventionEntry(
                            candidate_x_idx=int(generator.candidate_x_idx),
                            unit_size=cap,
                            max_repetitions=max_reps,
                            asset_type="generator",
                            asset_name=str(generator.name),
                            asset_order=int(generator.order),
                        )
                    )

        for order, cap_p in iv.storage_capacities_p.items():
            storage = fleet_static.storages[order]
            mb_p = float(storage.max_build_p[year_idx])
            if cap_p > 0:
                max_reps_p = int(math.floor(mb_p / cap_p))
                if max_reps_p > 0:
                    entries.append(
                        _InterventionEntry(
                            candidate_x_idx=int(storage.candidate_p_x_idx),
                            unit_size=cap_p,
                            max_repetitions=max_reps_p,
                            asset_type="storage_power",
                            asset_name=str(storage.name),
                            asset_order=int(storage.order),
                        )
                    )

        # Variable-duration storage only: energy is an independent decision variable
        for order, cap_e in iv.storage_capacities_e.items():
            storage = fleet_static.storages[order]
            if int(storage.duration[year_idx]) == 0 and cap_e > 0:
                mb_e = float(storage.max_build_e[year_idx])
                max_reps_e = int(math.floor(mb_e / cap_e))
                if max_reps_e > 0:
                    entries.append(
                        _InterventionEntry(
                            candidate_x_idx=int(storage.candidate_e_x_idx),
                            unit_size=cap_e,
                            max_repetitions=max_reps_e,
                            asset_type="storage_energy",
                            asset_name=str(storage.name),
                            asset_order=int(storage.order),
                        )
                    )

    return entries


# ---------------------------------------------------------------------------
# Intervention-set enumeration
# ---------------------------------------------------------------------------


def _enumerate_multisets(
    n_types: int,
    max_total: int,
    max_per_type: NDArray,
    current: List[int],
    idx: int,
    remaining: int,
    out: List[List[int]],
) -> None:
    """
    Recursive helper: enumerate all non-zero multisets up to *max_total* size.

    Parameters:
    -------
    n_types (int): Total number of intervention types.
    max_total (int): Maximum total interventions in a set (intervention_set_size).
    max_per_type (NDArray): Per-type repetition cap (max_repetitions).
    current (List[int]): Count vector being built (length n_types).
    idx (int): Current type index being filled.
    remaining (int): Remaining capacity in this set.
    out (List[List[int]]): Accumulator for completed sets.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Appends completed count vectors to out.
    """
    if idx == n_types:
        if sum(current) > 0:
            out.append(list(current))
        return

    cap = min(int(max_per_type[idx]), remaining)
    for count in range(0, cap + 1):
        current[idx] = count
        _enumerate_multisets(n_types, max_total, max_per_type, current, idx + 1, remaining - count, out)
    current[idx] = 0


def _generate_intervention_sets(
    interventions: List[_InterventionEntry],
    intervention_set_size: int,
    temp_dir: str,
) -> str:
    """
    Enumerate all valid intervention multisets and write them to a temp .npy file.

    Each row of the output array is an integer count vector: element i is the
    number of times intervention i appears in that set.  Only non-zero sets are
    included (at least one intervention must be present).

    Parameters:
    -------
    interventions (List[_InterventionEntry]): Available interventions for the year.
    intervention_set_size (int): Maximum total interventions per set.
    temp_dir (str): Directory for temporary files.

    Returns:
    -------
    str: File path of the written count-matrix .npy file.
    """
    n = len(interventions)
    max_per_type = np.array([iv.max_repetitions for iv in interventions], dtype=np.int32)

    sets: List[List[int]] = []
    _enumerate_multisets(n, intervention_set_size, max_per_type, [0] * n, 0, intervention_set_size, sets)

    count_matrix = np.array(sets, dtype=np.uint8)
    out_path = os.path.join(temp_dir, "intervention_sets.npy")
    np.save(out_path, count_matrix, allow_pickle=False)
    get_logger().info("Generated %d intervention sets for %d intervention types.", len(sets), n)
    return out_path


# ---------------------------------------------------------------------------
# Cost-space projection
# ---------------------------------------------------------------------------


def _compute_unit_annualised_costs(
    interventions: List[_InterventionEntry],
    fleet_static: Fleet_InstanceType,
    year_idx: int,
    year_count: int,
) -> NDArray:
    """
    Compute the annualised build cost of adding exactly one unit of each intervention.

    Uses the same NPV formula as ltcosts_m.calculate_annualised_build:
        present_value = (1 - (1 + r)^(-L)) / r
        annualised_cost = year_count * unit_size * 1e6 * capex / present_value

    Parameters:
    -------
    interventions (List[_InterventionEntry]): ScenarioInterventions for the current year.
    fleet_static (Fleet_InstanceType): Static fleet data.
    year_idx (int): Year index (0-based).
    year_count (int): Total years in the modelling horizon.

    Returns:
    -------
    NDArray: Float64 array of per-unit annualised costs, one per intervention.
    """
    costs = np.zeros(len(interventions), dtype=np.float64)

    generator_lookup = {int(g.order): g for g in fleet_static.generators.values()}
    storage_lookup = {int(s.order): s for s in fleet_static.storages.values()}

    for i, iv in enumerate(interventions):
        if iv.asset_type == "generator":
            asset = generator_lookup[iv.asset_order]
            unit_cost = asset.cost[year_idx]
        else:
            asset = storage_lookup[iv.asset_order]
            unit_cost = asset.cost[year_idx]

        r = float(unit_cost.discount_rate)
        lifetime = float(unit_cost.lifetime)

        if r > 1e-9 and lifetime > 1e-9:
            present_value = (1.0 - (1.0 + r) ** (-lifetime)) / r
        else:
            present_value = 0.0

        if present_value < 1e-6:
            costs[i] = 0.0
            continue

        if iv.asset_type == "generator" or iv.asset_type == "storage_power":
            costs[i] = year_count * iv.unit_size * 1e6 * float(unit_cost.capex_p) / present_value
        else:  # storage_energy
            costs[i] = year_count * iv.unit_size * 1e6 * float(unit_cost.capex_e) / present_value

    return costs


def _convert_to_cost_space(
    sets_path: str,
    unit_costs: NDArray,
    temp_dir: str,
) -> str:
    """
    Project each intervention set from count space to cost space.

    The cost vector for a set is: count_vector * unit_costs.
    This gives a fixed-length representation of each set in cost space,
    enabling clustering across sets of different composition.

    Parameters:
    -------
    sets_path (str): Path to the count-matrix .npy file.
    unit_costs (NDArray): Per-intervention unit annualised costs.
    temp_dir (str): Directory for temporary files.

    Returns:
    -------
    str: File path of the written cost-matrix .npy file.
    """
    count_matrix = np.load(sets_path, allow_pickle=False).astype(np.float64)
    cost_matrix = count_matrix * unit_costs[np.newaxis, :]
    out_path = os.path.join(temp_dir, "cost_vectors.npy")
    np.save(out_path, cost_matrix, allow_pickle=False)
    return out_path


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _find_medoid_indices(
    cost_matrix: NDArray,
    labels: NDArray,
    centroids: NDArray,
) -> NDArray:
    """
    Find the medoid (closest member by Manhattan distance) of each cluster.

    Parameters:
    -------
    cost_matrix (NDArray): Shape (n_sets, n_interventions) float64 cost vectors.
    labels (NDArray): Cluster label per row, shape (n_sets,).
    centroids (NDArray): Cluster centroid coordinates, shape (n_clusters, n_interventions).

    Returns:
    -------
    NDArray: Int64 array of row indices into cost_matrix, one per cluster.
    """
    n_clusters = centroids.shape[0]
    medoid_indices = np.zeros(n_clusters, dtype=np.int64)

    for k in range(n_clusters):
        member_mask = labels == k
        member_indices = np.where(member_mask)[0]
        if len(member_indices) == 0:
            medoid_indices[k] = 0
            continue
        distances = np.sum(np.abs(cost_matrix[member_indices] - centroids[k]), axis=1)
        medoid_indices[k] = int(member_indices[np.argmin(distances)])

    return medoid_indices


def _cluster_intervention_sets(
    costs_path: str,
    cluster_count: int,
) -> NDArray:
    """
    Cluster intervention sets in cost space using mini-batch k-means and return medoid indices.

    If the number of sets is at most cluster_count, all sets are returned as
    medoids (no clustering required).

    Parameters:
    -------
    costs_path (str): Path to the cost-matrix .npy file.
    cluster_count (int): Number of clusters (medoids) to return.

    Returns:
    -------
    NDArray: Int64 array of row indices into the sets file.
    """
    cost_matrix = np.load(costs_path, allow_pickle=False)

    if len(cost_matrix) <= cluster_count:
        return np.arange(len(cost_matrix), dtype=np.int64)

    kmeans = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(cost_matrix)
    return _find_medoid_indices(cost_matrix, labels, kmeans.cluster_centers_)


# ---------------------------------------------------------------------------
# Applying interventions to x
# ---------------------------------------------------------------------------


def _apply_intervention_set_to_x(
    base_x: NDArray,
    count_row: NDArray,
    interventions: List[_InterventionEntry],
    lower_bounds: NDArray,
    upper_bounds: NDArray,
) -> NDArray:
    """
    Apply a count-vector intervention set to a base x vector, then clip to bounds.

    Parameters:
    -------
    base_x (NDArray): Starting candidate-x vector (new build amounts for this year).
    count_row (NDArray): Integer count vector, one element per intervention type.
    interventions (List[_InterventionEntry]): Intervention definitions.
    lower_bounds (NDArray): Per-variable lower bounds for this year.
    upper_bounds (NDArray): Per-variable upper bounds for this year.

    Returns:
    -------
    NDArray: New x vector after applying the intervention set, clipped to bounds.
    """
    new_x = base_x.copy()
    for i, iv in enumerate(interventions):
        new_x[iv.candidate_x_idx] += int(count_row[i]) * iv.unit_size
    return np.clip(new_x, lower_bounds, upper_bounds)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pathway_planning(
    config: ModelConfig,
    parameters_static: ScenarioParameters_InstanceType,
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
    scenario_name: str,
    results_directory: str,
    lower_bounds: NDArray,
    upper_bounds: NDArray,
    initial_x_candidate: Optional[NDArray],
    initial_population: NDArray = "latinhypercube",  # type: ignore[assignment]
    interventions: Optional[ScenarioScenarioInterventions] = None,
) -> OptimizeResult:
    """
    Run the pathway planning optimisation process.

    For each investment step year (from parameters_static.investment_steps):

    1. Generates discrete interventions from the ScenarioInterventions object, one entry per
       asset build step defined in interventions.csv.
    2. Enumerates all valid multisets of 1..intervention_set_size interventions,
       respecting per-asset max_build limits.  Sets are written to a temp file.
    3. Projects each set to cost space (annualised build cost per intervention type)
       and clusters using mini-batch k-means.
    4. Takes the medoid (closest member by Manhattan distance) of each cluster as
       the representative set for that cluster.
    5. Evaluates each medoid set by applying it to the best x from the previous
       investment step and running unit commitment for the investment-step year.
    6. Advances the best solution to the next investment step (vintage tracking).

    Parameters:
    -------
    config (ModelConfig): Model configuration parameters.
    parameters_static (ScenarioParameters_InstanceType): Static scenario parameters.
    fleet_static (Fleet_InstanceType): Static fleet data.
    network_static (Network_InstanceType): Static network data.
    scenario_name (str): Name of the scenario being evaluated.
    results_directory (str): Directory to save results to.
    lower_bounds (NDArray[np.float64]): Array of lower bounds for decision variables,
        shape (year_count, n_vars).
    upper_bounds (NDArray[np.float64]): Array of upper bounds for decision variables,
        shape (year_count, n_vars).
    initial_x_candidate (NDArray[np.float64] | None): Optional initial candidate
        solution used as the starting configuration for the first investment step.
    initial_population (NDArray[np.float64] | str): Unused; kept for calling-convention
        parity with run_capacity_expansion.
    interventions (ScenarioInterventions | None): ScenarioInterventions container for this scenario.
        When None, no discrete interventions are available and each investment step
        evaluates only the base configuration.

    Returns:
    -------
    OptimizeResult: Result containing the best x vector and objective from the final
        investment step.
    """
    scenario_root = os.path.join(results_directory, f"{scenario_name}_pathway_planning")
    os.makedirs(scenario_root, exist_ok=True)
    temp_dir = os.path.join(scenario_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Vintage tracking — same structure as capacity_expansion
    generator_vintages: _VintageList = [[] for _ in range(len(fleet_static.generators))]
    storage_power_vintages: _VintageList = [[] for _ in range(len(fleet_static.storages))]
    storage_energy_vintages: _VintageList = [[] for _ in range(len(fleet_static.storages))]
    line_vintages: _VintageList = [[] for _ in range(len(network_static.major_lines))]

    records = _initialise_pathway_records(fleet_static, network_static)
    previous_best_x: Optional[NDArray] = initial_x_candidate.copy() if initial_x_candidate is not None else None
    final_result: Optional[OptimizeResult] = None

    for year in parameters_static.investment_steps:
        year_idx = int(year - parameters_static.first_year)
        first_t, last_t = static_m.get_year_t_boundaries(parameters_static, year_idx)

        (
            generator_existing,
            storage_power_existing,
            storage_energy_existing,
            major_line_existing,
            minor_line_existing,
        ) = _prepare_existing_capacity_arrays(
            fleet_static,
            network_static,
            int(year),
            year_idx,
            generator_vintages,
            storage_power_vintages,
            storage_energy_vintages,
            line_vintages,
        )

        raw_lower = lower_bounds[year_idx].astype(np.float64, copy=True)
        raw_upper = upper_bounds[year_idx].astype(np.float64, copy=True)
        year_lower, year_upper = _build_effective_year_bounds(
            fleet_static,
            network_static,
            year_idx,
            raw_lower,
            raw_upper,
            generator_existing,
            storage_power_existing,
            storage_energy_existing,
            major_line_existing,
        )

        base_x = previous_best_x if previous_best_x is not None else year_lower.copy()

        get_logger().info(
            "Pathway planning investment step %s: %d variables over intervals [%s, %s).",
            int(year),
            len(year_lower),
            first_t,
            last_t,
        )

        # --- Intervention generation, enumeration and clustering ---
        year_entries = _generate_interventions_for_year(year_idx, fleet_static, interventions)

        step_start = time.time()

        if not year_entries:
            get_logger().info(
                "Investment step %s: no interventions available — evaluating base configuration.", int(year)
            )
            best_solution = evaluate_capacity_expansion_solution(
                base_x,
                parameters_static,
                fleet_static,
                network_static,
                config.balancing_type,
                config.fixed_costs_threshold,
                first_t,
                last_t,
                year_idx,
                generator_existing,
                storage_power_existing,
                storage_energy_existing,
                major_line_existing,
                minor_line_existing,
            )
            best_x = base_x
        else:
            unit_costs = _compute_unit_annualised_costs(
                year_entries,
                fleet_static,
                year_idx,
                int(parameters_static.year_count),
            )
            sets_path = _generate_intervention_sets(
                year_entries,
                config.intervention_set_size,
                temp_dir,
            )
            costs_path = _convert_to_cost_space(sets_path, unit_costs, temp_dir)
            medoid_indices = _cluster_intervention_sets(costs_path, config.intervention_set_medoids_per_year)

            count_matrix = np.load(sets_path, allow_pickle=False)
            best_objective = float("inf")
            best_solution = None
            best_x = base_x.copy()

            get_logger().info(
                "Investment step %s: evaluating %d medoid intervention sets.",
                int(year),
                len(medoid_indices),
            )

            for medoid_idx in medoid_indices:
                x = _apply_intervention_set_to_x(
                    base_x,
                    count_matrix[medoid_idx],
                    year_entries,
                    year_lower,
                    year_upper,
                )
                solution = evaluate_capacity_expansion_solution(
                    x,
                    parameters_static,
                    fleet_static,
                    network_static,
                    config.balancing_type,
                    config.fixed_costs_threshold,
                    first_t,
                    last_t,
                    year_idx,
                    generator_existing,
                    storage_power_existing,
                    storage_energy_existing,
                    major_line_existing,
                    minor_line_existing,
                )
                objective = solution.lcoe + solution.penalties
                if objective < best_objective:
                    best_objective = objective
                    best_solution = solution
                    best_x = x

        step_end = time.time()

        get_logger().info(
            "Investment step %s complete in %.3f seconds. LCOE=%.6f, penalties=%.6f.",
            int(year),
            step_end - step_start,
            best_solution.lcoe,
            best_solution.penalties,
        )

        _update_vintages(
            best_solution,
            int(year),
            year_idx,
            generator_vintages,
            storage_power_vintages,
            storage_energy_vintages,
            line_vintages,
        )
        _append_pathway_records(records, int(year), best_solution)
        records["metrics"].append(
            {
                "year": int(year),
                "objective": float(best_solution.lcoe + best_solution.penalties),
                "lcoe": float(best_solution.lcoe),
                "penalties": float(best_solution.penalties),
                "solve_time_seconds": float(step_end - step_start),
            }
        )

        previous_best_x = best_x
        final_result = OptimizeResult(
            x=best_x,
            fun=float(best_solution.lcoe + best_solution.penalties),
            nit=0,
            success=best_solution.penalties < 1.0,
            message="Pathway planning complete.",
        )

    shutil.rmtree(temp_dir, ignore_errors=True)
    _write_pathway_records(records, scenario_root)

    return final_result if final_result is not None else OptimizeResult()