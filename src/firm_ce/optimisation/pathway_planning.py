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

from firm_ce.common.constants import DEBUG, NUM_THREADS, INTERVENTION_CHUNK_SIZE, PENALTY_MULTIPLIER
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
from firm_ce.optimisation.single_time import Solution, parallel_wrapper

""" # ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
# Typed alias used by the vintage helpers imported from capacity_expansion
_VintageList = List[List[Tuple[int, float, int]]] """

def generate_interventions_for_step(
    year_idx: int,
    scenario_interventions: Optional[ScenarioInterventions],
) -> Dict[int, Intervention]:
    """
    Build the dict of available discrete interventions for a given year. An intervention is
    determined to be available if the max build limit for all associated assets is not
    binding.

    Parameters:
    -------
    year_idx (int): Year index (0-based offset from first_year).
    scenario_interventions (ScenarioInterventions | None): ScenarioInterventions container for the
        scenario. Returns an empty dict when None.

    Returns:
    -------
    Dict[int, Intervention]: Available interventions for the year, keyed by intervention index.
    """
    if scenario_interventions is None:
        return {}

    return {
        idx: intervention
        for idx, intervention in scenario_interventions.interventions.get(year_idx, {}).items()
        if intervention.maximum_repetitions > 0
    }


def stream_combinations_to_bin(
    combinations_inst: combinations_with_replacement,
    step_interventions: Dict[int, Intervention],
    out_path: str,
    chunk_size: int = 100000,
) -> int:
    """
    Filter a combinations iterator against per-intervention repetition limits and write
    valid entries to a binary flat file.

    Each valid entry is a row of uint8 intervention indices. Rows are batched within
    each chunk and flushed once per chunk to minimise file-open overhead. The file is
    opened in binary-append mode so the function may be called incrementally.

    Parameters:
    -------
    combinations_inst (combinations_with_replacement): Iterator yielding combination tuples.
    step_interventions (Dict[int, Intervention]): Available interventions keyed by index,
        used to check maximum_repetitions constraints.
    out_path (str): Destination binary file path.
    chunk_size (int): Number of combinations to draw from the iterator per iteration.

    Returns:
    -------
    int: Number of valid combinations written.
    """
    valid_combinations = 0
    while True:
        chunk = np.array(list(islice(combinations_inst, chunk_size)), dtype=np.uint8)
        if len(chunk) == 0:
            break

        valid_entries = [
            entry
            for entry in chunk
            if all(
                step_interventions[idx].maximum_repetitions >= idx_count
                for idx, idx_count in Counter(entry).items()
            )
        ]

        if valid_entries:
            with open(out_path, "ab") as f:
                np.array(valid_entries, dtype=np.uint8).tofile(f)
            valid_combinations += len(valid_entries)

    return valid_combinations


def export_bin_to_csv(bin_path: str, n_cols: int, dtype: type) -> None:
    """
    Convert a binary flat file to a human-readable CSV equivalent.

    Reads the binary data from ``bin_path``, reshapes it to ``(-1, n_cols)``, and writes
    a matching ``.csv`` file in the same directory. The CSV format is inferred from
    ``dtype``: integer dtypes use ``"%d"``; floating-point dtypes use ``"%g"``.

    Intended to be called only when ``DEBUG`` is ``True``.

    Parameters:
    -------
    bin_path (str): Path to the source binary file.
    n_cols (int): Number of columns per row in the binary file.
    dtype (type): Numpy dtype used when the binary file was written (e.g. ``np.uint8``,
        ``np.float64``).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Writes a ``.csv`` file alongside ``bin_path`` with the same stem.
    """
    fmt = "%d" if np.issubdtype(dtype, np.integer) else "%g"
    data = np.fromfile(bin_path, dtype=dtype).reshape(-1, n_cols)
    csv_path = os.path.splitext(bin_path)[0] + ".csv"
    np.savetxt(csv_path, data, fmt=fmt, delimiter=",")


def generate_intervention_sets(
    step_interventions: Dict[int, Intervention],
    intervention_set_size: int,
    intervention_sets_dir: str,
) -> int:
    """
    Enumerate all valid intervention sets and write them to per-size CSV files.

    Each row of an output file is a combination of intervention indices (with
    repetition allowed up to each intervention's maximum_repetitions). Only sets
    that satisfy all repetition constraints are written.

    Parameters:
    -------
    step_interventions (Dict[int, Intervention]): Available interventions for the year,
        keyed by intervention index.
    intervention_set_size (int): Maximum total interventions per set.
    intervention_sets_dir (str): Directory for output CSV files.

    Returns:
    -------
    int: Total number of valid intervention sets generated.
    """
    intervention_sets_total_len = 0
    for n in range(1, intervention_set_size + 1):
        out_path = os.path.join(intervention_sets_dir, f"intervention_sets_{n}.bin")
        intervention_sets_n = combinations_with_replacement(step_interventions, n)
        intervention_sets_total_len += stream_combinations_to_bin(intervention_sets_n, step_interventions, out_path, INTERVENTION_CHUNK_SIZE)
        if DEBUG:
            export_bin_to_csv(out_path, n, np.uint8)

    return intervention_sets_total_len


def generate_capacity_and_cost_vectors(
    intervention_sets_dir: str,
    temp_dir: str,
    vector_len: int,
    step_interventions: Dict[int, Intervention],
    fleet: Fleet_InstanceType,
) -> Tuple[str, str]:
    """
    Project each intervention set from count space to capacity and cost space.

    For each intervention set file, builds a count matrix (rows = sets, cols = intervention
    indices) and multiplies by pre-computed per-intervention delta arrays to produce dense
    capacity and cost vectors of length vector_len.

    Parameters:
    -------
    intervention_sets_dir (str): Directory containing per-size intervention set .bin files.
    temp_dir (str): Directory for temporary output files.
    vector_len (int): Length of the candidate x-vector (number of decision variables).
    step_interventions (Dict[int, Intervention]): Available interventions keyed by index.
    fleet (Fleet_InstanceType): Static fleet data used to resolve x-vector positions.

    Returns:
    -------
    Tuple[str, str]: Paths to the (capacities, costs) binary files.
    """
    costs_path = os.path.join(temp_dir, "cost_vectors.bin")
    capacities_path = os.path.join(temp_dir, "capacities_vectors.bin")

    max_intervention_idx = max(step_interventions.keys())
    cost_delta_per_intervention = np.zeros((max_intervention_idx + 1, vector_len), dtype=np.float64)
    capacity_delta_per_intervention = np.zeros((max_intervention_idx + 1, vector_len), dtype=np.float64)

    for intervention_idx, intervention in step_interventions.items():
        generator_annualised_costs, storage_p_annualised_costs, storage_e_annualised_costs = intervention.annualised_costs
        generator_capacities, storage_capacities_p, storage_capacities_e = intervention.capacities

        for order, cost in generator_annualised_costs.items():
            cost_delta_per_intervention[intervention_idx, fleet.generators[order].candidate_x_idx] += cost
        for order, cost in storage_p_annualised_costs.items():
            cost_delta_per_intervention[intervention_idx, fleet.storages[order].candidate_p_x_idx] += cost
        for order, cost in storage_e_annualised_costs.items():
            cost_delta_per_intervention[intervention_idx, fleet.storages[order].candidate_e_x_idx] += cost

        for order, capacity in generator_capacities.items():
            capacity_delta_per_intervention[intervention_idx, fleet.generators[order].candidate_x_idx] += capacity
        for order, capacity_p in storage_capacities_p.items():
            capacity_delta_per_intervention[intervention_idx, fleet.storages[order].candidate_p_x_idx] += capacity_p
        for order, capacity_e in storage_capacities_e.items():
            capacity_delta_per_intervention[intervention_idx, fleet.storages[order].candidate_e_x_idx] += capacity_e

    sorted_intervention_set_filenames = sorted(
        (f for f in os.listdir(intervention_sets_dir) if f.endswith(".bin")),
        key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]),
    )
    for intervention_sets_filename in sorted_intervention_set_filenames:
        n_cols = int(os.path.splitext(intervention_sets_filename)[0].split("_")[-1])
        intervention_sets_path = os.path.join(intervention_sets_dir, intervention_sets_filename)
        intervention_sets = np.fromfile(intervention_sets_path, dtype=np.uint8).reshape(-1, n_cols)

        chunk_start = 0
        while chunk_start < len(intervention_sets):
            chunk = intervention_sets[chunk_start : chunk_start + INTERVENTION_CHUNK_SIZE]
            chunk_start += INTERVENTION_CHUNK_SIZE

            row_indices = np.arange(len(chunk))
            intervention_count_matrix = np.zeros((len(chunk), max_intervention_idx + 1), dtype=np.float64)
            for col_idx in range(n_cols):
                intervention_count_matrix[row_indices, chunk[:, col_idx]] += 1.0

            cost_vectors = intervention_count_matrix @ cost_delta_per_intervention
            capacities_vectors = intervention_count_matrix @ capacity_delta_per_intervention

            with open(costs_path, "ab") as f_costs:
                cost_vectors.tofile(f_costs)
            with open(capacities_path, "ab") as f_capacities:
                capacities_vectors.tofile(f_capacities)

    if DEBUG:
        export_bin_to_csv(costs_path, vector_len, np.float64)
        export_bin_to_csv(capacities_path, vector_len, np.float64)

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
        cluster_k_member_indices = np.where(cluster_assignment_indices == k)[0]
        if len(cluster_k_member_indices) == 0:
            medoid_indices[k] = 0
            continue
        member_distances_from_centroid = np.sum(np.abs(cost_matrix[cluster_k_member_indices] - centroids[k]), axis=1)
        medoid_indices[k] = int(cluster_k_member_indices[np.argmin(member_distances_from_centroid)])
    
    return np.sort(medoid_indices)


def cluster_intervention_sets(
    costs_path: str,
    vector_len: int,
    cluster_count: int,
) -> NDArray:
    """
    Cluster intervention sets in cost space using mini-batch k-means and return medoid indices.

    If the number of sets is at most cluster_count, all sets are returned as
    medoids (no clustering required).

    Parameters:
    -------
    costs_path (str): Path to the cost-matrix binary file (float64, shape (-1, vector_len)).
    vector_len (int): Number of columns in the cost matrix (length of the candidate x vector).
    cluster_count (int): Number of clusters (medoids) to return.

    Returns:
    -------
    NDArray: Int64 array of row indices into the cost matrix, one per cluster.
    """
    cost_matrix = np.fromfile(costs_path, dtype=np.float64).reshape(-1, vector_len)

    if len(cost_matrix) <= cluster_count:
        return np.arange(len(cost_matrix), dtype=np.int64)

    kmeans = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, n_init="auto", batch_size=256 * NUM_THREADS)
    cluster_assignment_indices = kmeans.fit_predict(cost_matrix)
    return find_medoid_indices(cost_matrix, cluster_assignment_indices, kmeans.cluster_centers_)


def select_new_build_capacities(
    medoid_indices: NDArray,
    capacities_path: str,
    vector_len: int,
    step_medoids_dir: str,
    step_year: int,
) -> NDArray:
    """
    Extract the capacity vectors for the selected medoid intervention sets.

    Parameters:
    -------
    medoid_indices (NDArray): Row indices into the capacities matrix identifying the medoids.
    capacities_path (str): Path to the capacities-matrix binary file (float64, shape (-1, vector_len)).
    vector_len (int): Number of columns in the capacities matrix (length of the candidate x vector).
    step_medoids_dir (str): Directory to write the per-year medoid capacities CSV.
    step_year (int): Investment step year; used to name the output file.

    Returns:
    -------
    NDArray: Float64 array of shape (n_medoids, vector_len) containing new-build capacity vectors.

    Side-effects:
    -------
    Writes medoid capacity vectors to ``<step_medoids_dir>/<step_year>.csv``.
    """
    capacities_matrix = np.memmap(capacities_path, dtype=np.float64, mode='r').reshape(-1, vector_len)
    medoid_new_build_capacities = np.array(capacities_matrix[medoid_indices])
    out_path = os.path.join(step_medoids_dir, f"{step_year}.bin")
    medoid_new_build_capacities.tofile(out_path)
    if DEBUG:
        export_bin_to_csv(out_path, vector_len, np.float64)
    return medoid_new_build_capacities


def generate_pathways(
    step_year: int,
    previous_step_year: int | None,
    step_feasible_dir: str,
    step_medoid_ids: NDArray,
) -> NDArray:
    """
    Build the pathway array for the current investment step.

    For the first step, returns a 2D array of shape (M, 1) containing each medoid ID as a
    single-element pathway. For subsequent steps, returns the cross-product of all previous
    pathways with the new medoid IDs, shape (P*M, S+1), where P is the number of previous
    pathways and S their column width.

    Parameters:
    -------
    step_year (int): Current investment step year.
    previous_step_year (int | None): Previous investment step year, or None for the first step.
    step_feasible_dir (str): Directory containing per-year feasible pathway files.
    step_medoid_ids (NDArray): Integer medoid IDs available at this step, shape (M,).

    Returns:
    -------
    NDArray: Int64 array of shape (M, 1) for the first step, or (P*M, S+1) for subsequent steps.

    Side-effects:
    -------
    Writes ``candidate_pathway_ids.npy`` to ``<step_feasible_dir>/<step_year>/``. When DEBUG
    is True, also writes ``candidate_pathway_ids.csv``.
    """
    current_step_dir = os.path.join(step_feasible_dir, str(step_year))
    os.makedirs(current_step_dir, exist_ok=True)
    current_step_pathways_path = os.path.join(current_step_dir, "candidate_pathway_ids.npy")

    if previous_step_year is not None:
        previous_step_pathways_path = os.path.join(step_feasible_dir, str(previous_step_year), "candidate_pathway_ids.npy")
        previous_step_pathways = np.load(previous_step_pathways_path)

        expanded_previous = np.repeat(previous_step_pathways, len(step_medoid_ids), axis=0)
        tiled_new_ids = np.tile(step_medoid_ids, len(previous_step_pathways))
        current_step_pathways = np.column_stack([expanded_previous, tiled_new_ids])
    else:
        current_step_pathways = np.asarray(step_medoid_ids, dtype=np.int64).reshape(-1, 1)

    np.save(current_step_pathways_path, current_step_pathways)
    if DEBUG:
        csv_path = os.path.splitext(current_step_pathways_path)[0] + ".csv"
        np.savetxt(csv_path, current_step_pathways, fmt="%d", delimiter=",")
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


def save_pathway_step(
    step_year: int,
    step_feasible_dir: str,
    pathway_medoid_ids: NDArray,
    lcoe: float,
    penalties: float,
) -> None:
    """
    Append a feasible pathway record to the per-year binary result files.

    Parameters:
    -------
    step_year (int): Investment step year; used to name the output sub-directory.
    step_feasible_dir (str): Root directory for feasible pathway results.
    pathway_medoid_ids (NDArray): 1-D int64 array of medoid IDs forming this pathway,
        one entry per investment step up to and including step_year.
    lcoe (float): LCOE of the evaluated solution [$/MWh].
    penalties (float): Soft-constraint penalty value of the evaluated solution.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Appends one row to ``<step_feasible_dir>/<step_year>/feasible_pathway_ids.bin`` and
    one row to ``<step_feasible_dir>/<step_year>/feasible_objectives.bin``.
    """
    step_path = os.path.join(step_feasible_dir, str(step_year))
    os.makedirs(step_path, exist_ok=True)

    pathway_medoid_ids_path = os.path.join(step_path, "feasible_pathway_ids.bin")
    objective_path = os.path.join(step_path, "feasible_objectives.bin")

    with open(pathway_medoid_ids_path, "ab") as f:
        pathway_medoid_ids.astype(np.int64).tofile(f)
    with open(objective_path, "ab") as f:
        np.array([lcoe, penalties], dtype=np.float64).tofile(f)

    return None


def build_x_existing(fleet_static: Fleet_InstanceType, year_idx: int, vector_len: int) -> NDArray:
    """
    Build the existing-capacity vector at a given year index.

    Accumulates the initial_capacity increments for each generator and the
    initial_power_capacity / initial_energy_capacity increments for each storage, from
    year 0 up to and including year_idx. Positive values represent additions; negative
    values represent retirements.

    Parameters:
    -------
    fleet_static (Fleet_InstanceType): Static fleet containing Generator and Storage instances.
    year_idx (int): Year index (0-based offset from first_year).
    vector_len (int): Length of the candidate x-vector.

    Returns:
    -------
    NDArray: Float64 array of shape (vector_len,) with cumulative existing capacity at year_idx.
    """
    x_existing = np.zeros(vector_len, dtype=np.float64)
    for generator in fleet_static.generators.values():
        if generator.candidate_x_idx < 0:
            continue
        for y in range(year_idx + 1):
            if y in generator.initial_capacity:
                x_existing[generator.candidate_x_idx] += generator.initial_capacity[y]
    for storage in fleet_static.storages.values():
        if storage.candidate_p_x_idx >= 0:
            for y in range(year_idx + 1):
                if y in storage.initial_power_capacity:
                    x_existing[storage.candidate_p_x_idx] += storage.initial_power_capacity[y]
        if storage.candidate_e_x_idx >= 0:
            for y in range(year_idx + 1):
                if y in storage.initial_energy_capacity:
                    x_existing[storage.candidate_e_x_idx] += storage.initial_energy_capacity[y]
    return x_existing


def parallel_wrapper_with_progress(
    xs: NDArray,
    parameters_static: ScenarioParameters_InstanceType,
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
    balancing_type: str,
    fixed_costs_threshold: float,
    first_t: int,
    last_t: int,
    chunk_size: int = 10000,
) -> NDArray:
    """
    Evaluate candidate solutions in chunks, logging progress every chunk.

    Splits xs column-wise into chunks of at most chunk_size columns, calls
    parallel_wrapper on each chunk, and concatenates the results. A progress log
    line is emitted after each chunk is processed.

    Parameters:
    -------
    xs (NDArray): Shape (vector_len, n_points) float64 matrix of candidate solutions.
    parameters_static (ScenarioParameters_InstanceType): Static scenario parameters.
    fleet_static (Fleet_InstanceType): Static fleet data.
    network_static (Network_InstanceType): Static network data.
    balancing_type (str): Balancing mode (e.g. 'full').
    fixed_costs_threshold (float): Upper bound on fixed cost intensity [$/MWh].
    first_t (int): First interval index (inclusive) for unit commitment.
    last_t (int): Last interval index (exclusive) for unit commitment.
    chunk_size (int): Maximum number of solutions evaluated per parallel_wrapper call.

    Returns:
    -------
    NDArray: Float64 array of shape (3, n_points). Row 0 is total energy (LCOE + penalties),
        row 1 is LCOE, row 2 is penalties.
    """
    n_points = xs.shape[1]
    result = np.zeros((3, n_points), dtype=np.float64)
    evaluated = 0
    while evaluated < n_points:
        chunk_end = min(evaluated + chunk_size, n_points)
        chunk_result = parallel_wrapper(
            xs[:, evaluated:chunk_end],
            parameters_static, fleet_static, network_static,
            balancing_type, fixed_costs_threshold, first_t, last_t,
        )
        result[:, evaluated:chunk_end] = chunk_result
        evaluated = chunk_end
        get_logger().info("    Evaluated %d / %d solutions.", evaluated, n_points)
    return result


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
    5. Evaluates all candidate pathways - each being the cross-product of medoid sets
       chosen at every preceding investment step plus the current step - by running unit
       commitment for the investment-step year.
    6. Advances every feasible pathway to the next investment step; infeasible pathways
       are discarded.

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
    logger = get_logger()
    n_steps = len(parameters_static.investment_steps)
    logger.info("Starting pathway planning for scenario '%s' (%d investment steps).", scenario_name, n_steps)

    scenario_root, temp_dir, intervention_sets_dir, step_medoids_dir, step_feasible_dir = initialise_pathway_planning_directory(scenario_name, results_directory)
    vector_len = len(lower_bounds)
    investment_steps_completed: List[int] = []
    previous_step_year = None
    final_result = OptimizeResult()

    for step_idx, year in enumerate(parameters_static.investment_steps):
        step_start = time.time()
        year_idx = int(year - parameters_static.first_year)
        first_t, last_t = static_m.get_year_t_boundaries(parameters_static, year_idx)
        logger.info("Investment step %d/%d: year %d.", step_idx + 1, n_steps, year)

        logger.info("  Generating interventions for year %d.", year)
        step_interventions = generate_interventions_for_step(year_idx, interventions)
        logger.info("  Year %d: %d interventions available.", year, len(step_interventions))

        logger.info("  Generating intervention sets (max size %d).", config.intervention_set_size)
        intervention_sets_total_len = generate_intervention_sets(step_interventions, config.intervention_set_size, intervention_sets_dir)
        get_logger().info("  Generated %d intervention sets.", intervention_sets_total_len)

        logger.info("  Building capacity and cost vectors for year %d.", year)
        capacities_path, costs_path = generate_capacity_and_cost_vectors(
            intervention_sets_dir, temp_dir, vector_len, step_interventions, fleet_static
        )
        logger.info("  Capacity and cost vectors built for year %d.", year)

        logger.info("  Clustering intervention sets (target %d medoids).", config.intervention_set_medoids_per_year)
        medoid_indices = cluster_intervention_sets(costs_path, vector_len, config.intervention_set_medoids_per_year)
        logger.info("  Year %d: selected %d medoid intervention sets.", year, len(medoid_indices))

        medoid_new_build_capacities = select_new_build_capacities(medoid_indices, capacities_path, vector_len, step_medoids_dir, year)

        step_medoid_ids = np.arange(medoid_new_build_capacities.shape[0], dtype=np.int64)
        pathway_medoid_ids = generate_pathways(year, previous_step_year, step_feasible_dir, step_medoid_ids)

        x_existing = build_x_existing(fleet_static, year_idx, vector_len)

        previous_step_capacity_arrays = [
            np.fromfile(os.path.join(step_medoids_dir, f"{prev_year}.bin"), dtype=np.float64).reshape(-1, vector_len)
            for prev_year in investment_steps_completed
        ]

        n_pathways = pathway_medoid_ids.shape[0]
        logger.info("  Assembling %d candidate pathway x-vectors for year %d.", n_pathways, year)
        xs = np.zeros((vector_len, n_pathways), dtype=np.float64)
        for i in range(n_pathways):
            x_previous_build = np.zeros(vector_len, dtype=np.float64)
            for col, prev_caps in enumerate(previous_step_capacity_arrays):
                x_previous_build += prev_caps[pathway_medoid_ids[i, col]]
            x_new_build = medoid_new_build_capacities[pathway_medoid_ids[i, -1]]
            x_candidate = x_existing + x_previous_build + x_new_build
            if np.any(x_candidate < 0.0):
                negative_indices = np.where(x_candidate < 0.0)[0]
                raise ValueError(
                    f"Negative x_candidate values in pathway {i} at year {year}: "
                    f"indices {negative_indices.tolist()} with values {x_candidate[negative_indices].tolist()}"
                )
            xs[:, i] = x_candidate

        logger.info("  Evaluating %d pathways for year %d.", n_pathways, year)
        results = parallel_wrapper_with_progress(
            xs, parameters_static, fleet_static, network_static,
            config.balancing_type, config.fixed_costs_threshold, first_t, last_t,
        )
        logger.info("  Pathway evaluation complete for year %d.", year)

        best_year_lcoe = float("inf")
        best_year_x: Optional[NDArray] = None
        n_feasible = 0
        for i in range(n_pathways):
            lcoe = float(results[1, i])
            penalties = float(results[2, i])
            if penalties == 0.0:
                n_feasible += 1
                save_pathway_step(year, step_feasible_dir, pathway_medoid_ids[i, :], lcoe, penalties)
                if lcoe < best_year_lcoe:
                    best_year_lcoe = lcoe
                    best_year_x = xs[:, i].copy()
            else:
                logger.debug("Infeasible pathway %d at year %d (lcoe=%.2f, penalties=%.6f)", i, year, lcoe, penalties)

        if DEBUG and n_feasible > 0:
            step_path = os.path.join(step_feasible_dir, str(year))
            export_bin_to_csv(os.path.join(step_path, "feasible_pathway_ids.bin"), pathway_medoid_ids.shape[1], np.int64)
            export_bin_to_csv(os.path.join(step_path, "feasible_objectives.bin"), 2, np.float64)

        if best_year_x is not None:
            final_result = OptimizeResult(x=best_year_x, fun=best_year_lcoe)
            logger.info(
                "  Year %d complete - %d/%d pathways feasible, best LCOE %.2f $/MWh (%.1f s).",
                year, n_feasible, n_pathways, best_year_lcoe, time.time() - step_start,
            )
        else:
            logger.warning("  Year %d: no feasible pathways found.", year)

        investment_steps_completed.append(year)
        previous_step_year = year

    logger.info("Pathway planning complete for scenario '%s'.", scenario_name)
    return final_result