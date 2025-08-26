import numpy as np
from numpy.typing import NDArray

from firm_ce.common.jit_overload import njit
from firm_ce.fast_methods import network_m, static_m, node_m, generator_m
from firm_ce.common.typing import float64, int64
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType

#@njit
def prefix_sums(array_1d: NDArray[np.float64]):
    """
    Compute prefix sums for fast segment statistics.

    Returns
    -------
    cumulative_sum_values : np.ndarray
        Prefix sums of the values (length = n+1).
    cumulative_sum_of_squared_values : np.ndarray
        Prefix sums of the squared values (length = n+1).
    """
    cumulative_sum_values = np.concatenate(([0.0], np.cumsum(array_1d)))
    cumulative_sum_of_squared_values = np.concatenate(([0.0], np.cumsum(array_1d * array_1d)))
    return cumulative_sum_values, cumulative_sum_of_squared_values

#@njit
def seg_mean_sse(start_index: int,
                 end_index_exclusive: int,
                 cumulative_sum_values: np.ndarray,
                 cumulative_sum_of_squared_values: np.ndarray):
    """
    Compute the segment mean and SSE (sum of squared errors to the mean)
    for array[start_index:end_index_exclusive] using prefix sums.

    Returns
    -------
    segment_mean_value : float
    sse_to_segment_mean : float
    segment_length : int
    """
    segment_length = end_index_exclusive - start_index
    segment_sum = cumulative_sum_values[end_index_exclusive] - cumulative_sum_values[start_index]
    segment_sum_of_squares = (
        cumulative_sum_of_squared_values[end_index_exclusive]
        - cumulative_sum_of_squared_values[start_index]
    )
    segment_mean_value = segment_sum / segment_length
    sse_to_segment_mean = segment_sum_of_squares - (segment_sum * segment_sum) / segment_length
    return segment_mean_value, sse_to_segment_mean, segment_length

#@njit
def best_medoid_for_segment(array_1d: np.ndarray,
                            start_index: int,
                            end_index_exclusive: int,
                            segment_mean_value: float):
    """
    Among array_1d[start_index:end_index_exclusive], find the medoid under squared loss,
    i.e., the element closest to the segment mean. Also return the additional penalty
    incurred by using the medoid instead of the mean.

    Returns
    -------
    medoid_value : float
    additional_penalty_due_to_medoid : float
        Equals segment_length * (medoid_value - segment_mean_value)^2
    """
    segment_values = array_1d[start_index:end_index_exclusive]
    # Index of the element closest to the mean (argmin of absolute deviation)
    medoid_offset_index = np.argmin(np.abs(segment_values - segment_mean_value))
    medoid_value = float(segment_values[medoid_offset_index])

    segment_length = end_index_exclusive - start_index
    additional_penalty_due_to_medoid = segment_length * (medoid_value - segment_mean_value) ** 2
    return medoid_value, additional_penalty_due_to_medoid

#@njit
def hierarchically_cluster_consecutive_medoids(array_1d: NDArray[np.float64],
                                               number_of_clusters: int):
    """
    Optimal contiguous K-segmentation minimizing total SSE with
    per-segment representative constrained to a MEDOID (an actual data value).

    Returns
    -------
    segment_medoids : np.ndarray
        Length = number_of_clusters; medoid value for each contiguous segment in chronological order.
    segment_lengths : np.ndarray
        Length = number_of_clusters; number of items in each contiguous segment.
    """
    total_item_count = array_1d.size

    cumulative_sum_values, cumulative_sum_of_squared_values = prefix_sums(array_1d)

    # dynamic_programming_min_cost[k, j] = minimal total SSE to segment array_1d[:j] into k segments
    dynamic_programming_min_cost = np.full((number_of_clusters + 1, total_item_count + 1),
                                           np.inf, dtype=np.float64)
    # previous_cut_index[k, j] = index i where the last cut placed before j for the optimal k-segmentation
    previous_cut_index = np.full((number_of_clusters + 1, total_item_count + 1),
                                 -1, dtype=np.int32)
    # selected_medoid_value[k, j] = medoid value of the last segment (i:j) for the optimal k-segmentation
    selected_medoid_value = np.full((number_of_clusters + 1, total_item_count + 1),
                                    np.nan, dtype=np.float64)

    # Base case: exactly 1 segment covering [0:j]
    for end_position in range(1, total_item_count + 1):
        segment_mean_value, sse_to_segment_mean, _segment_length = seg_mean_sse(
            0, end_position, cumulative_sum_values, cumulative_sum_of_squared_values
        )
        medoid_value, additional_penalty_due_to_medoid = best_medoid_for_segment(
            array_1d, 0, end_position, segment_mean_value
        )
        dynamic_programming_min_cost[1, end_position] = sse_to_segment_mean + additional_penalty_due_to_medoid
        previous_cut_index[1, end_position] = 0
        selected_medoid_value[1, end_position] = medoid_value

    # Fill DP for k = 2..number_of_clusters
    for cluster_count in range(2, number_of_clusters + 1):
        # Need at least `cluster_count` items to form `cluster_count` non-empty segments
        for end_position in range(cluster_count, total_item_count + 1):
            best_total_cost_for_state = np.inf
            best_previous_index_for_state = -1
            best_medoid_value_for_state = np.nan

            # Try the last cut at candidate_cut_index, where the previous part has (cluster_count-1) segments
            for candidate_cut_index in range(cluster_count - 1, end_position):
                segment_mean_value, sse_to_segment_mean, _segment_length = seg_mean_sse(
                    candidate_cut_index, end_position, cumulative_sum_values, cumulative_sum_of_squared_values
                )
                medoid_value, additional_penalty_due_to_medoid = best_medoid_for_segment(
                    array_1d, candidate_cut_index, end_position, segment_mean_value
                )
                candidate_total_cost = (
                    dynamic_programming_min_cost[cluster_count - 1, candidate_cut_index]
                    + sse_to_segment_mean
                    + additional_penalty_due_to_medoid
                )

                if candidate_total_cost < best_total_cost_for_state:
                    best_total_cost_for_state = candidate_total_cost
                    best_previous_index_for_state = candidate_cut_index
                    best_medoid_value_for_state = medoid_value

            dynamic_programming_min_cost[cluster_count, end_position] = best_total_cost_for_state
            previous_cut_index[cluster_count, end_position] = best_previous_index_for_state
            selected_medoid_value[cluster_count, end_position] = best_medoid_value_for_state

    # Backtrack to recover cut positions and medoids
    cut_positions = [total_item_count]
    medoid_values_chronological = []
    cluster_count = number_of_clusters
    end_position = total_item_count

    while cluster_count > 0:
        start_position = int(previous_cut_index[cluster_count, end_position])
        cut_positions.append(start_position)
        medoid_values_chronological.append(float(selected_medoid_value[cluster_count, end_position]))
        cluster_count, end_position = cluster_count - 1, start_position

    cut_positions.sort()
    medoid_values_chronological.reverse()  # chronological order

    # Compute lengths for each contiguous segment
    segment_lengths = [end_pos - start_pos for start_pos, end_pos in zip(cut_positions[:-1], cut_positions[1:])]

    return (np.asarray(medoid_values_chronological, dtype=np.float64),
            np.asarray(segment_lengths, dtype=np.int64))

#@njit
def chronological_time_period_clustering(residual_load: NDArray[np.float64],
                                         resolution: float,
                                         intervals_count: int,
                                         blocks_per_day: int,):
    intervals_per_day = int(round(24.0 / resolution))
    number_of_days = int(intervals_count // intervals_per_day)

    fitted_values = np.zeros(number_of_days * blocks_per_day, dtype=np.float64)
    block_lengths = np.zeros(fitted_values.shape, dtype=np.int64)

    for day in range(number_of_days):
        first_interval = day * intervals_per_day
        last_interval = first_interval + intervals_per_day
        day_residual_load = residual_load[first_interval:last_interval].copy()

        medoids, weights = hierarchically_cluster_consecutive_medoids(day_residual_load, blocks_per_day)

        first_block = day * blocks_per_day
        last_block = first_block + blocks_per_day
        fitted_values[first_block:last_block] = medoids
        block_lengths[first_block:last_block] = weights

    return fitted_values, block_lengths

#@njit
def get_block_intervals(block_lengths: NDArray[np.int64]):
    block_final_intervals = np.cumsum(block_lengths, dtype=np.int64)
    block_first_intervals = np.concatenate(([0], block_final_intervals[:-1]))
    return block_first_intervals, block_final_intervals

#@njit
def CTPC_datafiles(network: Network_InstanceType, 
                   fleet: Fleet_InstanceType, 
                   block_first_intervals: NDArray[np.int64], 
                   block_final_intervals: NDArray[np.int64]):
    for node in network.nodes.values():
        node_m.convert_full_to_simple(node, block_first_intervals, block_final_intervals)

    for generator in fleet.generators.values():
        generator_m.convert_full_to_simple(generator, block_first_intervals, block_final_intervals)
    return None

#@njit
def convert_full_to_simple(network: Network_InstanceType, 
                           fleet: Fleet_InstanceType,
                           static: ScenarioParameters_InstanceType,
                           blocks_per_day: int,):
    net_residual_load = network_m.calculate_net_residual_load(network)
    _, static.block_lengths = chronological_time_period_clustering(net_residual_load, static.resolution, static.intervals_count, blocks_per_day)
    block_first_intervals, block_final_intervals = get_block_intervals(static.block_lengths)
    CTPC_datafiles(network, fleet, block_first_intervals, block_final_intervals)
    static_m.set_block_resolutions(static, static.block_lengths)
    static.intervals_count = len(static.block_lengths)
    static_m.set_year_first_block(static, blocks_per_day)
    return None

@njit
def data_medoids_for_blocks(data_array: float64[:], block_first_intervals: int64[:], block_final_intervals: int64[:]):
    if len(data_array) == 0:
        return np.empty(0, dtype=np.float64)

    clustered_medoid_data_array = np.zeros(block_first_intervals.size, dtype=np.float64)

    for block_index, (start_idx, end_idx) in enumerate(zip(block_first_intervals, block_final_intervals)):
        block_data = data_array[start_idx:end_idx]
        block_mean = np.mean(block_data)
        medoid_offset = int(np.argmin(np.abs(block_data - block_mean)))
        block_medoid = block_data[medoid_offset]

        clustered_medoid_data_array[block_index] = block_medoid

    return clustered_medoid_data_array
    