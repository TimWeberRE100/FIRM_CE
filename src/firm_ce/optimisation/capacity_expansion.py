import numpy as np
from numpy.typing import NDArray

from firm_ce.optimisation.single_time import Solution
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType


def run_capacity_expansion(
    parameters_static: ScenarioParameters_InstanceType,
    fleet_static: Fleet_InstanceType,
    network_static: Network_InstanceType,
    balancing_type: str,
    fixed_costs_threshold: float,
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> None:
    """
    Iterate over each year in the modelling horizon, creating and evaluating a Solution instance for each year.

    Notes:
    -------
    - The candidate solution vector x is currently set to lower_bounds. In future, a dedicated function will
    generate the x value for each year based on capacity decisions from prior years.
    - Each Solution is evaluated independently, covering only the interval range [first_t, last_t) for the
    corresponding year.

    Parameters:
    -------
    parameters_static (ScenarioParameters_InstanceType): Static scenario parameters.
    fleet_static (Fleet_InstanceType): Static Fleet jitclass instance used to derive a dynamic copy for evaluation.
    network_static (Network_InstanceType): Static Network jitclass instance used to derive a dynamic copy for evaluation.
    balancing_type (str): Balancing mode (e.g., 'full' for balancing with the complete time-series over the
        entire time horizon at the specified resolution).
    fixed_costs_threshold (float): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
        low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit
        committment problem.
    lower_bounds (NDArray[np.float64]): Lower bounds of the decision variable vector, shape (year_count, n_vars).
        Row `year` is used as the candidate solution x for that year until a dedicated x-generation function is
        implemented.

    Returns:
    -------
    None.
    """
    for year in range(parameters_static.year_count):
        first_t = int(parameters_static.year_first_t[year])
        last_t = (
            int(parameters_static.year_first_t[year + 1])
            if year < parameters_static.year_count - 1
            else int(parameters_static.intervals_count)
        )

        x = lower_bounds[year].copy()  # Placeholder; will be replaced by a dedicated x-generation function

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
        solution.evaluate()
