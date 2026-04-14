import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS
from firm_ce.common.jit_overload import njit
from firm_ce.common.logging import get_logger
from firm_ce.common.typing import float64, int64, unicode_type
from firm_ce.fast_methods import static_m
from firm_ce.optimisation.single_time import Solution
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ModelConfig, ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType

if JIT_ENABLED:
    from numba import set_num_threads

    set_num_threads(int(NUM_THREADS))


Vintage = Tuple[int, float, int]


@njit
def apply_existing_capacity_deltas(
    fleet_instance: Fleet_InstanceType,
    network_instance: Network_InstanceType,
    year_idx: int64,
    generator_existing: float64[:],
    storage_power_existing: float64[:],
    storage_energy_existing: float64[:],
    major_line_existing: float64[:],
    minor_line_existing: float64[:],
) -> None:
    """
    Apply capacity carried forward from prior optimisation years to a dynamic solution.

    Opening baseline capacity is taken from the first model year only. This helper adds
    surviving endogenous builds from previous optimisation years without treating them as
    new builds in the current year.
    """
    for line in network_instance.major_lines.values():
        line.capacity += major_line_existing[line.order]

    for line in network_instance.minor_lines.values():
        line.capacity += minor_line_existing[line.order]

    for generator in fleet_instance.generators.values():
        carried_capacity = generator_existing[generator.order]
        generator.capacity += carried_capacity

        if carried_capacity > 0.0 and len(generator.data) > 0:
            generator.node.residual_load[:] -= generator.data * carried_capacity

    for storage in fleet_instance.storages.values():
        carried_power = storage_power_existing[storage.order]
        storage.power_capacity += carried_power

        if storage.duration[year_idx] > 0:
            storage.energy_capacity += carried_power * storage.duration[year_idx]
        else:
            storage.energy_capacity += storage_energy_existing[storage.order]


@njit
def parallel_capacity_expansion_wrapper(
    xs: float64[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: float64,
    first_t: int64,
    last_t: int64,
    year_idx: int64,
    generator_existing: float64[:],
    storage_power_existing: float64[:],
    storage_energy_existing: float64[:],
    major_line_existing: float64[:],
    minor_line_existing: float64[:],
) -> float64[:, :]:
    """
    Vectorised objective wrapper for yearly capacity-expansion solves.
    """
    n_points = xs.shape[1]
    result = np.zeros((3, n_points), dtype=np.float64)

    for j in range(n_points):
        xj = xs[:, j]
        sol = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold, first_t, last_t)
        apply_existing_capacity_deltas(
            sol.fleet,
            sol.network,
            year_idx,
            generator_existing,
            storage_power_existing,
            storage_energy_existing,
            major_line_existing,
            minor_line_existing,
        )
        sol.evaluate()
        result[0, j] = sol.lcoe + sol.penalties
        result[1, j] = sol.lcoe
        result[2, j] = sol.penalties

    return result


def evaluate_vectorised_capacity_expansion_xs(
    xs: float64[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: float64,
    first_t: int64,
    last_t: int64,
    year_idx: int64,
    generator_existing: float64[:],
    storage_power_existing: float64[:],
    storage_energy_existing: float64[:],
    major_line_existing: float64[:],
    minor_line_existing: float64[:],
) -> float64[:]:
    """
    Python wrapper for the yearly vectorised objective to expose timing information.
    """
    start_time = time.time()
    result = parallel_capacity_expansion_wrapper(
        xs,
        static,
        fleet,
        network,
        balancing_type,
        fixed_costs_threshold,
        first_t,
        last_t,
        year_idx,
        generator_existing,
        storage_power_existing,
        storage_energy_existing,
        major_line_existing,
        minor_line_existing,
    )
    end_time = time.time()
    print(f"Average yearly objective time: {(end_time - start_time) / max(xs.shape[1], 1):.4f} seconds.")
    print(f"Yearly iteration time: {(end_time - start_time):.4f} seconds.")
    return result[0, :]


def evaluate_capacity_expansion_solution(
    x: NDArray[np.float64],
    parameters_static: ScenarioParameters_InstanceType,
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
    balancing_type: str,
    fixed_costs_threshold: float,
    first_t: int,
    last_t: int,
    year_idx: int,
    generator_existing: NDArray[np.float64],
    storage_power_existing: NDArray[np.float64],
    storage_energy_existing: NDArray[np.float64],
    major_line_existing: NDArray[np.float64],
    minor_line_existing: NDArray[np.float64],
) -> Solution:
    """
    Evaluate a single yearly candidate and return the fully populated Solution.
    """
    solution = Solution(
        x,
        parameters_static,
        fleet_static,
        network_static,
        balancing_type,
        fixed_costs_threshold,
        first_t,
        last_t,
    )
    apply_existing_capacity_deltas(
        solution.fleet,
        solution.network,
        year_idx,
        generator_existing,
        storage_power_existing,
        storage_energy_existing,
        major_line_existing,
        minor_line_existing,
    )
    solution.evaluate()
    return solution


def _clip_initial_guess(
    x0: NDArray[np.float64] | None,
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    if x0 is None:
        return None
    if len(x0) != len(lower_bounds):
        return None
    return np.clip(np.asarray(x0, dtype=np.float64), lower_bounds, upper_bounds)


def _active_vintages(vintages: Sequence[Vintage], current_year: int) -> Tuple[List[Vintage], float]:
    active: List[Vintage] = []
    total = 0.0
    for build_year, capacity, lifetime in vintages:
        if lifetime <= 0 or (current_year - build_year) < lifetime:
            active.append((build_year, capacity, lifetime))
            total += capacity
    return active, total


def _prepare_existing_capacity_arrays(
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
    current_year: int,
    year_idx: int,
    generator_vintages: List[List[Vintage]],
    storage_power_vintages: List[List[Vintage]],
    storage_energy_vintages: List[List[Vintage]],
    line_vintages: List[List[Vintage]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    generator_existing = np.zeros(len(fleet_static.generators), dtype=np.float64)
    storage_power_existing = np.zeros(len(fleet_static.storages), dtype=np.float64)
    storage_energy_existing = np.zeros(len(fleet_static.storages), dtype=np.float64)
    major_line_existing = np.zeros(len(network_static.major_lines), dtype=np.float64)
    minor_line_existing = np.zeros(len(network_static.minor_lines), dtype=np.float64)

    for order, generator in fleet_static.generators.items():
        active, total = _active_vintages(generator_vintages[order], current_year)
        generator_vintages[order] = active
        generator_existing[order] = total
        minor_line_existing[generator.line.order] += total

    for order, storage in fleet_static.storages.items():
        active_power, total_power = _active_vintages(storage_power_vintages[order], current_year)
        storage_power_vintages[order] = active_power
        storage_power_existing[order] = total_power
        minor_line_existing[storage.line.order] += total_power

        if storage.duration[year_idx] == 0:
            active_energy, total_energy = _active_vintages(storage_energy_vintages[order], current_year)
            storage_energy_vintages[order] = active_energy
            storage_energy_existing[order] = total_energy
        else:
            storage_energy_vintages[order] = []

    for order, line in network_static.major_lines.items():
        active, total = _active_vintages(line_vintages[order], current_year)
        line_vintages[order] = active
        major_line_existing[order] = total

    return (
        generator_existing,
        storage_power_existing,
        storage_energy_existing,
        major_line_existing,
        minor_line_existing,
    )


def _effective_build_bounds(raw_min: float, raw_max: float, active_endogenous_capacity: float) -> Tuple[float, float]:
    """
    Convert gross annual build bounds into effective additional-build bounds.

    Gross min/max bounds describe the total endogenous capacity pathway allowed for a year.
    Surviving endogenous vintages therefore count against both bounds.
    """
    effective_min = max(0.0, float(raw_min) - active_endogenous_capacity)
    effective_max = max(0.0, float(raw_max) - active_endogenous_capacity)

    if effective_min > effective_max and np.isclose(effective_min, effective_max):
        effective_min = effective_max

    return effective_min, effective_max


def _build_effective_year_bounds(
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
    year_idx: int,
    raw_lower_bounds: NDArray[np.float64],
    raw_upper_bounds: NDArray[np.float64],
    generator_existing: NDArray[np.float64],
    storage_power_existing: NDArray[np.float64],
    storage_energy_existing: NDArray[np.float64],
    major_line_existing: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply retirement-aware headroom adjustments to the current year's build bounds.

    The first model year's initial capacity is treated as exogenous opening stock. Only
    active endogenous builds reduce the remaining min/max build headroom in later years.
    """
    effective_lower_bounds = raw_lower_bounds.copy()
    effective_upper_bounds = raw_upper_bounds.copy()
    cursor = 0

    for generator in fleet_static.generators.values():
        effective_lower_bounds[cursor], effective_upper_bounds[cursor] = _effective_build_bounds(
            raw_lower_bounds[cursor],
            raw_upper_bounds[cursor],
            generator_existing[generator.order],
        )
        cursor += 1

    for storage in fleet_static.storages.values():
        effective_lower_bounds[cursor], effective_upper_bounds[cursor] = _effective_build_bounds(
            raw_lower_bounds[cursor],
            raw_upper_bounds[cursor],
            storage_power_existing[storage.order],
        )
        cursor += 1

    for storage in fleet_static.storages.values():
        if storage.duration[year_idx] == 0:
            effective_lower_bounds[cursor], effective_upper_bounds[cursor] = _effective_build_bounds(
                raw_lower_bounds[cursor],
                raw_upper_bounds[cursor],
                storage_energy_existing[storage.order],
            )
        else:
            effective_lower_bounds[cursor] = 0.0
            effective_upper_bounds[cursor] = 0.0
        cursor += 1

    for line in network_static.major_lines.values():
        effective_lower_bounds[cursor], effective_upper_bounds[cursor] = _effective_build_bounds(
            raw_lower_bounds[cursor],
            raw_upper_bounds[cursor],
            major_line_existing[line.order],
        )
        cursor += 1

    return effective_lower_bounds, effective_upper_bounds


def _update_vintages(
    solution: Solution,
    current_year: int,
    year_idx: int,
    generator_vintages: List[List[Vintage]],
    storage_power_vintages: List[List[Vintage]],
    storage_energy_vintages: List[List[Vintage]],
    line_vintages: List[List[Vintage]],
) -> None:
    for generator in solution.fleet.generators.values():
        if generator.new_build <= 0:
            continue
        lifetime = int(generator.cost[year_idx].lifetime)
        generator_vintages[generator.order].append((current_year, float(generator.new_build), lifetime))

    for storage in solution.fleet.storages.values():
        if storage.new_build_p > 0:
            lifetime = int(storage.cost[year_idx].lifetime)
            storage_power_vintages[storage.order].append((current_year, float(storage.new_build_p), lifetime))
        if storage.duration[year_idx] == 0 and storage.new_build_e > 0:
            lifetime = int(storage.cost[year_idx].lifetime)
            storage_energy_vintages[storage.order].append((current_year, float(storage.new_build_e), lifetime))

    for line in solution.network.major_lines.values():
        if line.new_build <= 0:
            continue
        lifetime = int(line.cost[year_idx].lifetime)
        line_vintages[line.order].append((current_year, float(line.new_build), lifetime))


def _initialise_pathway_records(
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
) -> Dict[str, List[Dict[str, float]]]:
    generator_names = [generator.name for generator in fleet_static.generators.values()]
    storage_names = [storage.name for storage in fleet_static.storages.values()]
    line_names = [line.name for line in network_static.major_lines.values()]

    return {
        "generator_names": generator_names,
        "storage_names": storage_names,
        "line_names": line_names,
        "generators_build": [],
        "generators_capacity": [],
        "storages_power_build": [],
        "storages_power_capacity": [],
        "storages_energy_build": [],
        "storages_energy_capacity": [],
        "lines_build": [],
        "lines_capacity": [],
        "metrics": [],
    }


def _append_pathway_records(records: Dict[str, List[Dict[str, float]]], current_year: int, solution: Solution) -> None:
    year_record = {"year": current_year}

    generator_lookup = {generator.name: generator for generator in solution.fleet.generators.values()}
    records["generators_build"].append(
        {**year_record, **{name: float(generator_lookup[name].new_build) for name in records["generator_names"]}}
    )
    records["generators_capacity"].append(
        {**year_record, **{name: float(generator_lookup[name].capacity) for name in records["generator_names"]}}
    )

    storage_lookup = {storage.name: storage for storage in solution.fleet.storages.values()}
    records["storages_power_build"].append(
        {**year_record, **{name: float(storage_lookup[name].new_build_p) for name in records["storage_names"]}}
    )
    records["storages_power_capacity"].append(
        {**year_record, **{name: float(storage_lookup[name].power_capacity) for name in records["storage_names"]}}
    )
    records["storages_energy_build"].append(
        {**year_record, **{name: float(storage_lookup[name].new_build_e) for name in records["storage_names"]}}
    )
    records["storages_energy_capacity"].append(
        {**year_record, **{name: float(storage_lookup[name].energy_capacity) for name in records["storage_names"]}}
    )

    line_lookup = {line.name: line for line in solution.network.major_lines.values()}
    records["lines_build"].append(
        {**year_record, **{name: float(line_lookup[name].new_build) for name in records["line_names"]}}
    )
    records["lines_capacity"].append(
        {**year_record, **{name: float(line_lookup[name].capacity) for name in records["line_names"]}}
    )


def _write_pathway_records(records: Dict[str, List[Dict[str, float]]], scenario_root: str) -> None:
    out_dir = os.path.join(scenario_root, "pathway")
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(records["generators_build"]).to_csv(os.path.join(out_dir, "generators_new_build.csv"), index=False)
    pd.DataFrame(records["generators_capacity"]).to_csv(
        os.path.join(out_dir, "generators_cumulative_capacity.csv"), index=False
    )
    pd.DataFrame(records["storages_power_build"]).to_csv(
        os.path.join(out_dir, "storages_power_new_build.csv"), index=False
    )
    pd.DataFrame(records["storages_power_capacity"]).to_csv(
        os.path.join(out_dir, "storages_power_cumulative_capacity.csv"), index=False
    )
    pd.DataFrame(records["storages_energy_build"]).to_csv(
        os.path.join(out_dir, "storages_energy_new_build.csv"), index=False
    )
    pd.DataFrame(records["storages_energy_capacity"]).to_csv(
        os.path.join(out_dir, "storages_energy_cumulative_capacity.csv"), index=False
    )
    pd.DataFrame(records["lines_build"]).to_csv(os.path.join(out_dir, "lines_new_build.csv"), index=False)
    pd.DataFrame(records["lines_capacity"]).to_csv(
        os.path.join(out_dir, "lines_cumulative_capacity.csv"), index=False
    )
    pd.DataFrame(records["metrics"]).to_csv(os.path.join(out_dir, "capacity_expansion_metrics.csv"), index=False)


def run_capacity_expansion(
    config: ModelConfig,
    parameters_static: ScenarioParameters_InstanceType,
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
    scenario_name: str,
    results_directory: str,
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    initial_x_candidate: NDArray[np.float64] | None,
    initial_population: NDArray[np.float64] | str = "latinhypercube",
) -> OptimizeResult:
    """
    Solve capacity expansion sequentially, one modelling year at a time.

    This preserves the feature/capacity-expansion branch's multiyear input setup and
    full-horizon trace handling, but carries forward endogenous builds between yearly solves
    in the same spirit as the Indonesia repo's year-by-year pathway logic.
    """
    scenario_root = os.path.join(results_directory, f"{scenario_name}_capacity_expansion")
    os.makedirs(scenario_root, exist_ok=True)

    generator_vintages: List[List[Vintage]] = [[] for _ in range(len(fleet_static.generators))]
    storage_power_vintages: List[List[Vintage]] = [[] for _ in range(len(fleet_static.storages))]
    storage_energy_vintages: List[List[Vintage]] = [[] for _ in range(len(fleet_static.storages))]
    line_vintages: List[List[Vintage]] = [[] for _ in range(len(network_static.major_lines))]

    records = _initialise_pathway_records(fleet_static, network_static)
    previous_best_x = initial_x_candidate if initial_x_candidate is not None else None
    final_result: OptimizeResult | None = None

    for year_idx in range(parameters_static.year_count):
        current_year = int(parameters_static.first_year + year_idx)
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
            current_year,
            year_idx,
            generator_vintages,
            storage_power_vintages,
            storage_energy_vintages,
            line_vintages,
        )

        raw_lower_bounds = lower_bounds[year_idx].astype(np.float64, copy=True)
        raw_upper_bounds = upper_bounds[year_idx].astype(np.float64, copy=True)
        year_lower_bounds, year_upper_bounds = _build_effective_year_bounds(
            fleet_static,
            network_static,
            year_idx,
            raw_lower_bounds,
            raw_upper_bounds,
            generator_existing,
            storage_power_existing,
            storage_energy_existing,
            major_line_existing,
        )
        clipped_x0 = _clip_initial_guess(previous_best_x, year_lower_bounds, year_upper_bounds)

        get_logger().info(
            "Capacity expansion year %s: solving %s variables over intervals [%s, %s).",
            current_year,
            len(year_lower_bounds),
            first_t,
            last_t,
        )

        solve_start = time.time()
        if np.allclose(year_lower_bounds, year_upper_bounds):
            candidate_x = year_lower_bounds.copy()
            best_solution = evaluate_capacity_expansion_solution(
                candidate_x,
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
            final_result = OptimizeResult(
                x=candidate_x,
                fun=best_solution.lcoe + best_solution.penalties,
                nit=0,
                success=True,
                message="All yearly bounds fixed.",
            )
        else:
            result = differential_evolution(
                x0=clipped_x0,
                func=evaluate_vectorised_capacity_expansion_xs,
                bounds=list(zip(year_lower_bounds, year_upper_bounds)),
                args=(
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
                ),
                tol=0,
                maxiter=config.iterations,
                popsize=config.population,
                init=initial_population if year_idx == 0 else "latinhypercube",
                mutation=(0.2, config.mutation),
                recombination=config.recombination,
                disp=True,
                polish=False,
                updating="deferred",
                workers=1,
                vectorized=True,
            )
            best_solution = evaluate_capacity_expansion_solution(
                result.x.copy(),
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
            final_result = result

        solve_end = time.time()
        get_logger().info(
            "Capacity expansion year %s complete in %.3f seconds. LCOE=%.6f, penalties=%.6f.",
            current_year,
            solve_end - solve_start,
            best_solution.lcoe,
            best_solution.penalties,
        )

        _update_vintages(
            best_solution,
            current_year,
            year_idx,
            generator_vintages,
            storage_power_vintages,
            storage_energy_vintages,
            line_vintages,
        )
        _append_pathway_records(records, current_year, best_solution)
        records["metrics"].append(
            {
                "year": current_year,
                "objective": float(best_solution.lcoe + best_solution.penalties),
                "lcoe": float(best_solution.lcoe),
                "penalties": float(best_solution.penalties),
                "solve_time_seconds": float(solve_end - solve_start),
            }
        )
        previous_best_x = final_result.x.copy()

    _write_pathway_records(records, scenario_root)
    return final_result if final_result is not None else OptimizeResult()
