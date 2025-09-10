import time

import numpy as np

from ..common.constants import JIT_ENABLED, NUM_THREADS, PENALTY_MULTIPLIER
from ..common.jit_overload import jitclass, njit, prange
from ..common.typing import boolean, float64, unicode_type
from ..fast_methods import fleet_m, generator_m, line_m, network_m, static_m, storage_m
from ..system.components import Fleet_InstanceType
from ..system.parameters import ScenarioParameters_InstanceType
from ..system.topology import Network_InstanceType
from .balancing import balance_for_period

if JIT_ENABLED:
    from numba import set_num_threads

    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        ("x", float64[:]),
        ("evaluated", boolean),
        ("lcoe", float64),
        ("penalties", float64),
        ("balancing_type", unicode_type),
        ("fixed_costs_threshold", float64),
        # Static jitclass instances
        ("static", ScenarioParameters_InstanceType),
        # Dynamic jitclass instances
        ("fleet", Fleet_InstanceType),
        ("network", Network_InstanceType),
    ]
else:
    solution_spec = []


@jitclass(solution_spec)
class Solution:
    def __init__(
        self,
        x: float64[:],
        static: ScenarioParameters_InstanceType,
        fleet: Fleet_InstanceType,
        network: Network_InstanceType,
        balancing_type: unicode_type,
        fixed_costs_threshold: float64,
    ) -> None:
        self.x = x
        self.evaluated = False
        self.lcoe = 0.0
        self.penalties = 0.0

        # These are static jitclass instances. It is UNSAFE to modify these
        # within a worker process of the optimiser
        self.static = static
        self.balancing_type = balancing_type
        self.fixed_costs_threshold = fixed_costs_threshold

        # These are dynamic jitclass instances. It is SAFE to modify
        # some attributes within a worker process of the optimiser
        self.network = network_m.create_dynamic_copy(network)  # Includes static reference to data
        self.fleet = fleet_m.create_dynamic_copy(
            fleet, self.network.nodes, self.network.minor_lines
        )  # Includes static reference to data

        fleet_m.build_capacities(self.fleet, x, self.static.interval_resolutions)
        network_m.build_capacity(self.network, x)

        fleet_m.allocate_memory(self.fleet, self.static.intervals_count)
        network_m.allocate_memory(self.network, self.static.intervals_count)

        network_m.assign_storage_merit_orders(self.network, self.fleet.storages)
        network_m.assign_flexible_merit_orders(self.network, self.fleet.generators)

    def balance_residual_load(self) -> boolean:
        fleet_m.initialise_stored_energies(self.fleet)

        for year in range(self.static.year_count):
            first_t, last_t = static_m.get_year_t_boundaries(self.static, year)

            fleet_m.initialise_annual_limits(self.fleet, year, first_t)

            balance_for_period(first_t, last_t, self.balancing_type == "full", self, year)

            annual_unserved_energy = network_m.calculate_period_unserved_energy(
                self.network, first_t, last_t, self.static.interval_resolutions
            )

            # End early if reliability constraint breached for any year
            if not static_m.check_reliability_constraint(self.static, year, annual_unserved_energy):
                self.penalties += (self.static.year_count - year) * annual_unserved_energy * PENALTY_MULTIPLIER
                return False
        return True

    def calculate_fixed_costs(self) -> float64:
        total_costs = 0.0
        years_float = self.static.year_count * self.static.fom_scalar

        for generator in self.fleet.generators.values():
            total_costs += generator_m.calculate_fixed_costs(generator, years_float, self.static.year_count)

        for storage in self.fleet.storages.values():
            total_costs += storage_m.calculate_fixed_costs(storage, years_float, self.static.year_count)

        for line in self.network.major_lines.values():
            total_costs += line_m.calculate_fixed_costs(line, years_float, self.static.year_count)

        for line in self.network.minor_lines.values():
            total_costs += line_m.calculate_fixed_costs(line, years_float, self.static.year_count)

        return total_costs

    def calculate_variable_costs(self) -> float64:
        total_costs = 0.0

        fleet_m.calculate_lt_generations(
            self.fleet,
            self.static.interval_resolutions,
        )
        network_m.calculate_lt_flows(
            self.network,
            self.static.interval_resolutions,
        )

        for generator in self.fleet.generators.values():
            total_costs += generator_m.calculate_variable_costs(generator)

        for storage in self.fleet.storages.values():
            total_costs += storage_m.calculate_variable_costs(storage)

        for line in self.network.major_lines.values():
            total_costs += line_m.calculate_variable_costs(line)

        for line in self.network.minor_lines.values():
            total_costs += line_m.calculate_variable_costs(line)

        return total_costs

    def check_fixed_costs(self, fixed_costs: float64) -> boolean:
        return (fixed_costs / sum(self.static.year_energy_demand) / 1000) < self.fixed_costs_threshold  # $/MWh_demand

    def objective(self):
        total_costs = self.calculate_fixed_costs()
        if not self.check_fixed_costs(total_costs):
            return self.lcoe, total_costs * PENALTY_MULTIPLIER  # End early if fixed cost constraint breached

        reliability_check = self.balance_residual_load()
        if not reliability_check:
            return self.lcoe, self.penalties  # End early if reliability constraint breached

        total_costs += self.calculate_variable_costs()

        total_line_losses = network_m.calculate_lt_line_losses(self.network)

        lcoe = total_costs / np.abs(sum(self.static.year_energy_demand) - total_line_losses) / 1000  # $/MWh

        return lcoe, self.penalties

    def evaluate(self):
        self.lcoe, self.penalties = self.objective()
        self.evaluated = True
        return self


if JIT_ENABLED:
    Solution_InstanceType = Solution.class_type.instance_type
else:
    Solution_InstanceType = Solution


@njit(parallel=True)
def parallel_wrapper(
    xs: float64[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: float64,
) -> float64[:, :]:
    """
    parallel_wrapper, but also returns LCOE and penalty separately
    """
    n_points = xs.shape[1]
    result = np.zeros((3, n_points), dtype=np.float64)
    for j in prange(n_points):
        xj = xs[:, j]
        sol = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        sol.evaluate()
        result[0, j] = sol.lcoe + sol.penalties
        result[1, j] = sol.lcoe
        result[2, j] = sol.penalties
    return result


# @njit
def evaluate_vectorised_xs(
    xs: float64[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: float64,
):
    start_time = time.time()
    result = parallel_wrapper(xs, static, fleet, network, balancing_type, fixed_costs_threshold)
    end_time = time.time()
    print(f"Objective time: {NUM_THREADS*(end_time-start_time)/xs.shape[1]:.4f} seconds")
    print(f"Iteration time: {(end_time-start_time):.4f} seconds for {NUM_THREADS} workers")
    return result[0, :]
