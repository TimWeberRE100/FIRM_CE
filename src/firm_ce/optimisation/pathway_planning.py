import math
import os
import shutil
import time
from collections import namedtuple, Counter
from typing import List, Optional, Tuple, Dict
from itertools import combinations, islice, combinations_with_replacement

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from sklearn.cluster import MiniBatchKMeans

from firm_ce.common.constants import NP_INT64_MAX, INTERVENTION_CHUNK_SIZE
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
from firm_ce.system.interventions import ScenarioInterventions, Intervention
from firm_ce.system.parameters import ModelConfig, ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
# Typed alias used by the vintage helpers imported from capacity_expansion
_VintageList = List[List[Tuple[int, float, int]]]


# ---------------------------------------------------------------------------
# Intervention generation
# ---------------------------------------------------------------------------


def generate_interventions_for_step(
    year_idx: int,
    fleet_static: Fleet_InstanceType,
    scenario_interventions: Optional[ScenarioInterventions],
) -> List[Intervention]:
    """
    Build the list of available discrete interventions for a given year. An intervention is
    determined to be available if the max build limit for all associated assets is not 
    binding.

    Parameters:
    -------
    year_idx (int): Year index (0-based offset from first_year).
    fleet_static (Fleet_InstanceType): Static fleet data, used to read max_build limits.
    system_interventions (ScenarioInterventions | None): ScenarioInterventions container for the
        scenario. Returns an empty list when None.

    Returns:
    -------
    List[Intervention]: List of available interventions for the year.
    """
    if scenario_interventions is None:
        return []
    
    step_interventions = {}
    for idx, intervention in scenario_interventions.interventions.get(year_idx, {}).items():
        if intervention.maximum_repetitions > 0:
            step_interventions[idx] = intervention

    return step_interventions


def stream_combinations_to_csv(
        combinations_inst: combinations_with_replacement, 
        step_interventions: Dict[Intervention], 
        out_path: str, 
        chunk_size: int = 100000
    ) -> int:
    valid_combinations = 0
    while True:
        chunk = np.array(list(islice(combinations_inst, chunk_size)), dtype=np.uint8)
        if len(chunk) == 0:
            break
        
        for entry in chunk:
            tally = Counter(entry)
            if [step_interventions[idx].maximum_repetitions >= idx_count for idx, idx_count in tally.items()]:
                valid_combinations += 1
                print(entry)
                with open(out_path, 'a') as f:
                    np.savetxt(f, entry.reshape(1, -1), fmt='%d', delimiter=',')
    return valid_combinations


def generate_intervention_sets(
    step_interventions: List[Intervention],
    intervention_set_size: int,
    temp_dir: str,
) -> str:
    """
    Enumerate all valid intervention sets and write them to a temp .npy file.

    Each row of the output array is an integer count vector: element i is the
    number of times intervention i appears in that set.  Only non-zero sets are
    included (at least one intervention must be present).

    Parameters:
    -------
    interventions (List[Intervention]): Available interventions for the year.
    intervention_set_size (int): Maximum total interventions per set.
    temp_dir (str): Directory for temporary files.

    Returns:
    -------
    str: File path of the written count-matrix .npy file.
    """
    out_dir = os.path.join(temp_dir, f"intervention_sets")
    os.makedirs(out_dir, exist_ok=True)

    intervention_sets_total_len = 0
    for n in range(1, intervention_set_size + 1):
        out_path = os.path.join(out_dir, f"intervention_sets_{n}.csv")
        intervention_sets_n = combinations_with_replacement(list(step_interventions.keys()), n)
        intervention_sets_total_len += stream_combinations_to_csv(intervention_sets_n, step_interventions, out_path, INTERVENTION_CHUNK_SIZE)

    get_logger().info("Generated %d intervention sets.", intervention_sets_total_len)
    return out_dir


def convert_to_cost_space(
    intervention_sets_dir: str,
    temp_dir: str,
    cost_vector_len: int,
    step_interventions: Dict[Intervention],
    fleet: Fleet_InstanceType,
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
    out_path = os.path.join(temp_dir, "cost_vectors.csv")

    for intervention_sets_filename in os.listdir(intervention_sets_dir):
        intervention_sets_path = os.path.join(intervention_sets_dir,intervention_sets_filename)

        with open(intervention_sets_path, 'r') as f1:
            while True:
                chunk = np.genfromtxt(f1, delimiter=',', max_rows=INTERVENTION_CHUNK_SIZE)
                print(chunk.size)
                if chunk.size == 0:
                    break

                for row in range(chunk.shape[0]):
                    cost_vector = np.zeros(cost_vector_len, dtype = np.float64)

                    if len(chunk.shape) == 1:
                        intervention_idx = chunk[row]
                        generator_annualised_costs, storage_p_annualised_costs, storage_e_annualised_costs = step_interventions[intervention_idx].annualised_costs
                        for order, cost in generator_annualised_costs.items():
                            cost_vector[fleet.generators[order].candidate_x_idx] += cost
                        for order, cost in storage_p_annualised_costs.items():
                            cost_vector[fleet.storages[order].candidate_p_x_idx] += cost
                        for order, cost in storage_e_annualised_costs.items():
                            cost_vector[fleet.storages[order].candidate_e_x_idx] += cost
                    else:
                        for col in range(chunk.shape[1]):
                            intervention_idx = chunk[row,col]
                            generator_annualised_costs, storage_p_annualised_costs, storage_e_annualised_costs = step_interventions[intervention_idx].annualised_costs
                            for order, cost in generator_annualised_costs.items():
                                cost_vector[fleet.generators[order].candidate_x_idx] += cost
                            for order, cost in storage_p_annualised_costs.items():
                                cost_vector[fleet.storages[order].candidate_p_x_idx] += cost
                            for order, cost in storage_e_annualised_costs.items():
                                cost_vector[fleet.storages[order].candidate_e_x_idx] += cost
                    with open(out_path, 'a') as f2:
                        np.savetxt(f2, cost_vector.reshape(1, -1), fmt='%d', delimiter=',')

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

def initialise_pathway_planning_directory(scenario_name: str, results_directory: str) -> Tuple[str, str]:
    scenario_root = os.path.join(results_directory, f"{scenario_name}_pathway_planning")
    os.makedirs(scenario_root, exist_ok=True)
    temp_dir = os.path.join(scenario_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    return scenario_root, temp_dir

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
    interventions: Optional[ScenarioInterventions] = None,
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
    interventions (ScenarioInterventions | None): ScenarioInterventions container for this scenario.
        When None, no discrete interventions are available and each investment step
        evaluates only the base configuration.

    Returns:
    -------
    OptimizeResult: Result containing the best x vector and objective from the final
        investment step.
    """
    scenario_root, temp_dir = initialise_pathway_planning_directory(scenario_name, results_directory)
    
    generator_vintages: _VintageList = [[] for _ in range(len(fleet_static.generators))]
    storage_power_vintages: _VintageList = [[] for _ in range(len(fleet_static.storages))]
    storage_energy_vintages: _VintageList = [[] for _ in range(len(fleet_static.storages))]
    line_vintages: _VintageList = [[] for _ in range(len(network_static.major_lines))]

    #records = _initialise_pathway_records(fleet_static, network_static)
    #previous_best_x: Optional[NDArray] = initial_x_candidate.copy() if initial_x_candidate is not None else None
    #final_result: Optional[OptimizeResult] = None

    for year in parameters_static.investment_steps:
        year_idx = int(year - parameters_static.first_year)
        first_t, last_t = static_m.get_year_t_boundaries(parameters_static, year_idx)

        step_interventions = generate_interventions_for_step(year_idx, fleet_static, interventions)
        intervention_sets_dir = generate_intervention_sets(
            step_interventions,
            config.intervention_set_size,
            temp_dir,
        )
        costs_path = convert_to_cost_space(intervention_sets_dir, temp_dir, len(lower_bounds), step_interventions, fleet_static)
        medoid_indices = cluster_intervention_sets(costs_path, config.intervention_set_medoids_per_year)

        """"""

        # Create lists of capacities for current year + append new records to the vintage lists
        # Why are vintage lists important?
        # They are needed to select the active vintages - however, they are currently empty lists?
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

        # Lower and upper bounds are currently 1D arrays and cannot be indexed in this way
        # Need to also generate 2D arrays - 1 row per year - while the existing 1D array acts as a horizon-level limit
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
        #_append_pathway_records(records, int(year), best_solution)
        """ records["metrics"].append(
            {
                "year": int(year),
                "objective": float(best_solution.lcoe + best_solution.penalties),
                "lcoe": float(best_solution.lcoe),
                "penalties": float(best_solution.penalties),
                "solve_time_seconds": float(step_end - step_start),
            }
        ) """

        previous_best_x = best_x
        final_result = OptimizeResult(
            x=best_x,
            fun=float(best_solution.lcoe + best_solution.penalties),
            nit=0,
            success=best_solution.penalties < 1.0,
            message="Pathway planning complete.",
        )

    #shutil.rmtree(temp_dir, ignore_errors=True)
    #_write_pathway_records(records, scenario_root)

    return final_result if final_result is not None else OptimizeResult()