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

from firm_ce.common.constants import NUM_THREADS, INTERVENTION_CHUNK_SIZE, PENALTY_MULTIPLIER
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
from firm_ce.optimisation.single_time import Solution

""" # ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
# Typed alias used by the vintage helpers imported from capacity_expansion
_VintageList = List[List[Tuple[int, float, int]]] """

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
    intervention_sets_dir: str,
) -> None:
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
    intervention_sets_total_len = 0
    for n in range(1, intervention_set_size + 1):
        out_path = os.path.join(intervention_sets_dir, f"intervention_sets_{n}.csv")
        intervention_sets_n = combinations_with_replacement(list(step_interventions.keys()), n)
        intervention_sets_total_len += stream_combinations_to_csv(intervention_sets_n, step_interventions, out_path, INTERVENTION_CHUNK_SIZE)

    get_logger().info("Generated %d intervention sets.", intervention_sets_total_len)
    return None


def generate_capacity_and_cost_vectors(
    intervention_sets_dir: str,
    temp_dir: str,
    vector_len: int,
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
    costs_path = os.path.join(temp_dir, "cost_vectors.csv")
    capacities_path = os.path.join(temp_dir, "capacities_vectors.csv")

    for intervention_sets_filename in os.listdir(intervention_sets_dir):
        intervention_sets_path = os.path.join(intervention_sets_dir,intervention_sets_filename)

        with open(intervention_sets_path, 'r') as f1:
            while True:
                chunk = np.genfromtxt(f1, delimiter=',', max_rows=INTERVENTION_CHUNK_SIZE)
                
                if chunk.size == 0:
                    break

                for row in range(chunk.shape[0]):
                    cost_vector = np.zeros(vector_len, dtype = np.float64)
                    capacities_vector = np.zeros(vector_len, dtype = np.float64)

                    if len(chunk.shape) == 1:
                        intervention_idx = chunk[row]
                        generator_annualised_costs, storage_p_annualised_costs, storage_e_annualised_costs = step_interventions[intervention_idx].annualised_costs
                        for order, cost in generator_annualised_costs.items():
                            cost_vector[fleet.generators[order].candidate_x_idx] += cost
                        for order, cost in storage_p_annualised_costs.items():
                            cost_vector[fleet.storages[order].candidate_p_x_idx] += cost
                        for order, cost in storage_e_annualised_costs.items():
                            cost_vector[fleet.storages[order].candidate_e_x_idx] += cost

                        generator_capacities, storage_capacities_p, storage_capacities_e = step_interventions[intervention_idx].capacities
                        for order, capacity in generator_capacities.items():
                            capacities_vector[fleet.generators[order].candidate_x_idx] += capacity
                        for order, capacity in storage_capacities_p.items():
                            capacities_vector[fleet.storages[order].candidate_p_x_idx] += capacity
                        for order, capacity in storage_capacities_e.items():
                            capacities_vector[fleet.storages[order].candidate_e_x_idx] += capacity
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

                            generator_capacities, storage_capacities_p, storage_capacities_e = step_interventions[intervention_idx].capacities
                            for order, capacity in generator_capacities.items():
                                capacities_vector[fleet.generators[order].candidate_x_idx] += capacity
                            for order, capacity in storage_capacities_p.items():
                                capacities_vector[fleet.storages[order].candidate_p_x_idx] += capacity
                            for order, capacity in storage_capacities_e.items():
                                capacities_vector[fleet.storages[order].candidate_e_x_idx] += capacity

                    with open(costs_path, 'a') as f2:
                        np.savetxt(f2, cost_vector.reshape(1, -1), fmt='%d', delimiter=',')

                    with open(capacities_path, 'a') as f3:
                        np.savetxt(f3, capacities_vector.reshape(1, -1), fmt='%d', delimiter=',')

    return capacities_path, costs_path


def find_medoid_indices(
    cost_matrix: NDArray,
    cluster_assignment_indices: NDArray,
    centroids: NDArray,
) -> NDArray:
    """
    Find the medoid (closest member by Manhattan distance) of each cluster.

    Parameters:
    -------
    cost_matrix (NDArray): Shape (n_sets, n_interventions) float64 cost vectors.
    cluster_assignment_indices (NDArray): Cluster label per row, shape (n_sets,).
    centroids (NDArray): Cluster centroid coordinates, shape (n_clusters, n_interventions).

    Returns:
    -------
    NDArray: Int64 array of row indices into cost_matrix, one per cluster.
    """
    n_clusters = centroids.shape[0]
    medoid_indices = np.zeros(n_clusters, dtype=np.int64)

    for k in range(n_clusters):
        cluster_k_member_mask = cluster_assignment_indices == k
        cluster_k_member_indices = np.where(cluster_k_member_mask)[0]
        if len(cluster_k_member_indices) == 0:
            medoid_indices[k] = 0
            continue
        member_distances_from_centroid = np.sum(np.abs(cost_matrix[cluster_k_member_indices] - centroids[k]), axis=1)
        medoid_indices[k] = int(cluster_k_member_indices[np.argmin(member_distances_from_centroid)])
    
    return np.sort(medoid_indices)


def cluster_intervention_sets(
    costs_path: str,
    cluster_count: int,
) -> NDArray:
    """
    Cluster intervention sets in cost space using mini-batch k-means and return medoid indices.

    If the number of sets is at most cluster_count, all sets are returned as
    medoids (no clustering required).

    Parameters:
    -------
    costs_path (str): Path to the cost-matrix .csv file.
    cluster_count (int): Number of clusters (medoids) to return.

    Returns:
    -------
    NDArray: Int64 array of row indices into the sets file.
    """
    cost_matrix = np.genfromtxt(costs_path, delimiter=',') # This will be a problem if cost matrix is very large. Is there a way to actually do clustering in this case?

    if len(cost_matrix) <= cluster_count:
        return np.arange(len(cost_matrix), dtype=np.int64)

    kmeans = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, n_init="auto", batch_size=256*NUM_THREADS)
    cluster_assignment_indices = kmeans.fit_predict(cost_matrix)
    return find_medoid_indices(cost_matrix, cluster_assignment_indices, kmeans.cluster_centers_)


def select_new_build_capacities(medoid_indices: NDArray, capacities_path: str, vector_len: int, step_medoids_dir: str, step_year: int) -> NDArray:
    # NEED TO VALIDATE THERE IS NO OFF-BY-ONE ERROR IN THE INDEXING. Does it start at 0 or 1?
    medoid_iterator_jumps = np.insert(np.diff(medoid_indices) - 1, 0, medoid_indices[0])
    medoid_new_build_capacities = np.zeros((len(medoid_indices), vector_len), dtype=np.float64)
    out_path = os.path.join(step_medoids_dir, f"{step_year}.csv")

    with open(capacities_path, 'r') as f:
        for i, jump in enumerate(medoid_iterator_jumps):
            sliced_row = islice(f, jump, jump + 1)
            row_data = np.genfromtxt(sliced_row, delimiter=',')
            medoid_new_build_capacities[i] = row_data

    np.savetxt(out_path, medoid_new_build_capacities, delimiter=',')
    return medoid_new_build_capacities


def generate_pathways(step_year: int, previous_step_year: int | None, step_feasible_dir: str, step_medoid_ids: NDArray) -> NDArray:
    previous_step_pathways = np.array([], dtype=np.int64)
    current_step_pathways = step_medoid_ids
    previous_step_pathways_path = os.path.join(step_feasible_dir, previous_step_year, "pathway_medoid_ids.csv")
    current_step_pathways_path = os.path.join(step_feasible_dir, step_year, "pathway_medoid_ids.csv")

    if previous_step_year:
        previous_step_pathways = np.genfromtxt(previous_step_pathways_path, dtype=np.int64, delimiter=',')
        current_step_pathways = np.zeros((previous_step_pathways.shape[0] * len(step_medoid_ids), previous_step_pathways.shape[1] + 1), dtype=np.int64)
    
    current_step_pathway_id = 0
    for step_medoid_id in step_medoid_ids:
        for pathway in previous_step_pathways:
            current_step_pathway = np.append(pathway, step_medoid_id)
            current_step_pathways[current_step_pathway_id] = current_step_pathway
            current_step_pathway_id += 1

    np.savetxt(current_step_pathways_path, current_step_pathway, delimiter=',')
    return current_step_pathways


def initialise_pathway_planning_directory(scenario_name: str, results_directory: str) -> Tuple[str, str]:
    scenario_root = os.path.join(results_directory, f"{scenario_name}_pathway_planning")
    os.makedirs(scenario_root, exist_ok=True)

    temp_dir = os.path.join(scenario_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    intervention_sets_dir = os.path.join(temp_dir, "intervention_sets")
    os.makedirs(intervention_sets_dir, exist_ok=True)

    step_medoids_dir = os.path.join(scenario_root, "step_medoids")
    os.makedirs(step_medoids_dir, exist_ok=True)

    step_feasible_dir = os.path.join(scenario_root, "step_feasible")
    os.makedirs(step_feasible_dir, exist_ok=True)

    return scenario_root, temp_dir, intervention_sets_dir, step_medoids_dir, step_feasible_dir


def save_pathway_step(evaluated_solution: Solution, step_year: int, step_feasible_dir: str, pathway_medoid_ids: NDArray):
    step_path = os.path.join(step_feasible_dir, step_year)
    os.makedirs(step_path, exist_ok=True)

    pathway_medoid_ids_path = os.path.join(step_path, "pathway_medoid_ids.csv")
    objective_path = os.path.join(step_path, "objective.csv")

    with open(pathway_medoid_ids_path, 'a') as f:
        np.savetxt(f, pathway_medoid_ids, delimiter=',')
    with open(objective_path, 'a') as f:
        np.savetxt(f, np.array([evaluated_solution.lcoe, evaluated_solution.penalties]), delimiter=',')

    # Can add more data here in future, whatever is needed for the final results
    return None


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
    scenario_root, temp_dir, intervention_sets_dir, step_medoids_dir, step_feasible_dir = initialise_pathway_planning_directory(scenario_name, results_directory)
    previous_step_year = None

    """ generator_vintages: _VintageList = [[] for _ in range(len(fleet_static.generators))]
    storage_power_vintages: _VintageList = [[] for _ in range(len(fleet_static.storages))]
    storage_energy_vintages: _VintageList = [[] for _ in range(len(fleet_static.storages))]
    line_vintages: _VintageList = [[] for _ in range(len(network_static.major_lines))] """

    #records = _initialise_pathway_records(fleet_static, network_static)
    #previous_best_x: Optional[NDArray] = initial_x_candidate.copy() if initial_x_candidate is not None else None
    #final_result: Optional[OptimizeResult] = None
    
    for year in parameters_static.investment_steps:
        year_idx = int(year - parameters_static.first_year)
        first_t, last_t = static_m.get_year_t_boundaries(parameters_static, year_idx)

        step_interventions = generate_interventions_for_step(year_idx, fleet_static, interventions)
        generate_intervention_sets(
            step_interventions,
            config.intervention_set_size,
            intervention_sets_dir
        )
        capacities_path, costs_path = generate_capacity_and_cost_vectors(intervention_sets_dir, temp_dir, len(lower_bounds), step_interventions, fleet_static)
        medoid_indices = cluster_intervention_sets(costs_path, config.intervention_set_medoids_per_year)
        medoid_new_build_capacities = select_new_build_capacities(medoid_indices, capacities_path, len(lower_bounds), step_medoids_dir, year)
        pathway_medoid_ids = generate_pathways(year, previous_step_year, step_feasible_dir, range(0,medoid_new_build_capacities.shape[0] + 1))

        
        # Need to add a step to add a baseline "generation capacity" to x informed by the least cost pathway
        # Also, this needs to run in parallel
        for i in range(pathway_medoid_ids.shape[0]):
            x_existing = np.zeros(len(lower_bounds), dtype=np.float64) # Need to change this to be based on the initial_capacity added or retired each year
            x_previous_build = calculate_previous_build(pathway_medoid_ids[i,:], step_feasible_dir, previous_step_year)
            x_new_build = medoid_new_build_capacities[pathway_medoid_ids[i,-1]]
            x_candidate = x_existing + x_previous_build + x_new_build

            solution = Solution(
                x_candidate,
                parameters_static,
                fleet_static,
                network_static,
                config.balancing_type,
                config.fixed_costs_threshold,
                first_t,
                last_t,
            )
            solution.evaluate()

            if solution.lcoe < PENALTY_MULTIPLIER: # LCOE is currently not right, since it is only considering 1 year horizon
                save_pathway_step(solution, year, step_feasible_dir, pathway_medoid_ids)
            else:
                print(f"Infeasible {i}")      

        previous_step_year = year  

    return final_result if final_result is not None else OptimizeResult()