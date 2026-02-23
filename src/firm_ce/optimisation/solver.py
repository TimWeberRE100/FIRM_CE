import csv
import os
from itertools import chain
from logging import Logger
from typing import Dict, List, Tuple, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.optimisation.broad_optimum import (
    append_to_midpoint_csv,
    broad_optimum_objective,
    build_broad_optimum_var_info,
    create_groups_dict,
    create_midpoint_csv,
    read_broad_optimum_bands,
    write_broad_optimum_bands,
    write_broad_optimum_records,
)
from firm_ce.optimisation.single_time import Solution, evaluate_vectorised_xs
from firm_ce.system.components import Fleet_InstanceType, Generator_InstanceType, Storage_InstanceType
from firm_ce.system.parameters import ModelConfig, ScenarioParameters_InstanceType
from firm_ce.system.topology import Line_InstanceType, Network_InstanceType


class Solver:
    def __init__(
        self,
        config: ModelConfig,
        initial_x_candidate: NDArray[np.float64],
        parameters_static: ScenarioParameters_InstanceType,
        fleet_static: Fleet_InstanceType,
        network_static: Network_InstanceType,
        scenario_logger: Logger,
        scenario_name: str,
        initial_population: NDArray[np.float64] | str = "latinhypercube",
    ) -> None:
        """
        Initialise the Solver with the model configuration and static system data.

        Parameters:
        -------
        config (ModelConfig): Model configuration settings including optimisation
            type, population size, mutation and recombination rates, and near-optimum
            tolerance.
        initial_x_candidate (NDArray[np.float64]): Initial guess for the decision
            variable vector. Ignored (set to None) if the array is empty.
        parameters_static (ScenarioParameters_InstanceType): Static scenario
            parameters used during solution evaluation.
        fleet_static (Fleet_InstanceType): A static instance of the Fleet jitclass.
        network_static (Network_InstanceType): A static instance of the Network jitclass.
        scenario_logger (Logger): Logger instance for scenario-level messages.
        scenario_name (str): Name of the scenario, used for output file paths.
        initial_population (NDArray[np.float64] | str): Initial population
            for differential evolution. Either a 2-D array or a scipy initialisation
            strategy string (default "latinhypercube").

        Returns:
        -------
        None.

        Side-effects:
        -------
        Computes and stores decision variable bounds and broad optimum variable
        metadata.

        Exceptions:
        -------
        None.
        """
        self.config = config
        self.decision_x0 = initial_x_candidate if len(initial_x_candidate) > 0 else None
        self.parameters_static = parameters_static
        self.fleet_static = fleet_static
        self.network_static = network_static
        self.logger = scenario_logger
        self.lower_bounds, self.upper_bounds = self.get_bounds()
        self.broad_optimum_var_info = build_broad_optimum_var_info(fleet_static, network_static)
        self.scenario_name = scenario_name
        self.result = None
        self.optimal_lcoe = None
        self.initial_population = initial_population
        self.iterations = config.iterations

    def get_bounds(self) -> NDArray[np.float64]:
        """
        Compute lower and upper bounds for all decision variables.

        Bounds are assembled in the same order as the candidate solution vector:
        generator power capacities, storage power capacities, storage energy
        capacities, and major line power capacities.

        For storage energy capacity, the bound is set to 0 when a fixed duration
        ratio is specified (i.e. duration != 0), as the energy capacity is then
        derived from power capacity rather than optimised independently.

        Parameters:
        -------
        None.

        Returns:
        -------
        NDArray[np.float64]: A tuple of (lower_bounds, upper_bounds), each a 1-D
            array of length equal to the number of decision variables.

        Side-effects:
        -------
        None.

        Exceptions:
        -------
        None.
        """
        def power_capacity_bounds(
            asset_list: Union[List[Generator_InstanceType], List[Storage_InstanceType], List[Line_InstanceType]],
            build_cap_constraint: str,
        ) -> List[float]:
            return [getattr(asset, build_cap_constraint) for asset in asset_list]

        def energy_capacity_bounds(storage_list: List[Storage_InstanceType], build_cap_constraint: str) -> List[float]:
            return [getattr(s, build_cap_constraint) if s.duration == 0 else 0.0 for s in storage_list]

        generators = list(self.fleet_static.generators.values())
        storages = list(self.fleet_static.storages.values())
        lines = list(self.network_static.major_lines.values())

        lower_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "min_build"),
                    power_capacity_bounds(storages, "min_build_p"),
                    energy_capacity_bounds(storages, "min_build_e"),
                    power_capacity_bounds(lines, "min_build"),
                )
            )
        )

        upper_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "max_build"),
                    power_capacity_bounds(storages, "max_build_p"),
                    energy_capacity_bounds(storages, "max_build_e"),
                    power_capacity_bounds(lines, "max_build"),
                )
            )
        )

        return lower_bounds, upper_bounds

    def initialise_callback(self) -> None:
        """
        Create empty callback and population CSV files in the temp results directory.

        Parameters:
        -------
        None.

        Returns:
        -------
        None.

        Side-effects:
        -------
        Creates the results/temp directory if it does not exist and writes callback.csv, 
        population.csv, and population_energies.csvwith empty contents.

        Exceptions:
        -------
        None.
        """
        temp_dir = os.path.join("results", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "callback.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population_energies.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)

    def get_differential_evolution_args(
        self,
    ) -> Tuple[ScenarioParameters_InstanceType, Fleet_InstanceType, Network_InstanceType, str, float]:
        """
        Assemble the argument tuple passed to solution evaluation functions. Static instances are
        copied to form dynamic instances within each worker process of the differential evolution.

        Parameters:
        -------
        None.

        Returns:
        -------
        Tuple[ScenarioParameters_InstanceType, Fleet_InstanceType,
            Network_InstanceType, str, float]: A tuple of (parameters_static,
            fleet_static, network_static, balancing_type, fixed_costs_threshold).

        Side-effects:
        -------
        None.

        Exceptions:
        -------
        None.
        """
        args = (
            self.parameters_static,
            self.fleet_static,
            self.network_static,
            self.config.balancing_type,
            self.config.fixed_costs_threshold,
        )
        return args

    def run_differential_evolution(self, objective_function: Callable, args: Tuple) -> OptimizeResult:
        """
        Run scipy differential evolution with the solver's configuration.

        Parameters:
        -------
        objective_function (Callable): Vectorised objective function accepting a
            2-D candidate array and additional args.
        args (Tuple): Extra arguments forwarded to objective_function after the
            candidate array.

        Returns:
        -------
        OptimizeResult: The result object returned by scipy differential_evolution,
            containing the best solution vector and fitness value.

        Side-effects:
        -------
        Invokes the callback function at the end of each iteration, which appends
        the best solution to results/temp/callback.csv and optionally writes
        population files.

        Exceptions:
        -------
        None.
        """
        result = differential_evolution(
            x0=self.decision_x0,
            func=objective_function,
            bounds=list(zip(self.lower_bounds, self.upper_bounds)),
            args=args,
            tol=0,
            maxiter=self.iterations,
            popsize=self.config.population,
            init=self.initial_population,
            mutation=(0.2, self.config.mutation),
            recombination=self.config.recombination,
            disp=True,
            polish=False,
            updating="deferred",
            callback=callback,
            workers=1,
            vectorized=True,
        )
        return result

    def single_time(self) -> None:
        """
        Run a standard single-period cost minimisation using differential evolution.

        Initialises the callback output files and runs differential evolution with
        the vectorised evaluation function, storing the result in self.result.

        Parameters:
        -------
        None.

        Returns:
        -------
        None.

        Side-effects:
        -------
        Writes callback and population files to results/temp/. Stores the
        OptimizeResult in self.result.

        Exceptions:
        -------
        None.
        """
        self.initialise_callback()
        self.result = self.run_differential_evolution(evaluate_vectorised_xs, self.get_differential_evolution_args())

    def get_band_lcoe_max(self) -> float:
        """
        Evaluate the initial (optimal) solution and compute the near-optimum LCOE ceiling.

        The ceiling is set to the optimal LCOE scaled by (1 + near_optimal_tol) from
        the model config. A warning is logged if the initial guess carries a significant
        operational penalty, suggesting it may not actually represent a reliable solution.

        Parameters:
        -------
        None.

        Returns:
        -------
        float: Maximum allowable LCOE for a near-optimal solution.

        Side-effects:
        -------
        Stores the optimal LCOE in self.optimal_lcoe. Logs a warning if the initial
        guess has a penalty greater than 1.

        Exceptions:
        -------
        None.
        """
        solution = Solution(self.decision_x0, *self.get_differential_evolution_args())

        if solution.penalties > 1:
            self.logger.warning(
                f"Initial guess (assumed optimal solution) has a penalty of {solution.penalties}."
                f"It is recommended to double-check initial_guess.csv contains the correct optimal solution."
            )

        self.optimal_lcoe = solution.lcoe
        band_lcoe_max = self.optimal_lcoe * (1 + self.config.near_optimal_tol)

        return band_lcoe_max

    def find_near_optimal_band(self) -> Dict[str, Tuple[float]]:
        """
        Find the minimum and maximum feasible capacity for each near-optimum group.

        For each group of decision variables flagged for near-optimum exploration,
        runs two differential evolution optimisations to find the solution that
        minimises (or maximises) the summed group variable values while remaining
        within the near-optimum LCOE band. Feasible solutions encountered during
        the search are collected and written to CSV output files.

        Parameters:
        -------
        None.

        Returns:
        -------
        Dict[str, Tuple[float]]: Mapping from group key to a tuple of
            (min_candidate_x, max_candidate_x) solution vectors.

        Side-effects:
        -------
        Writes near_optimal_space.csv and near_optimal_bands.csv to the scenario's
        near_optimum results directory.

        Exceptions:
        -------
        None.
        """
        band_lcoe_max = self.get_band_lcoe_max()
        evaluation_records = []
        bands = {}
        groups = create_groups_dict(self.broad_optimum_var_info)

        for group_key, idx_list in groups.items():
            self.logger.info(f"[near_optimum] exploring group '{group_key}'")

            bands_record = []
            for band_type in ("min", "max"):
                match band_type:
                    case "min":
                        self.logger.info(f"[near_optimum] finding MIN for group '{group_key}'")
                    case "max":
                        self.logger.info(f"[near_optimum] finding MAX for group '{group_key}'")

                args = (
                    self.get_differential_evolution_args(),
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    band_type,
                )

                result = self.run_differential_evolution(broad_optimum_objective, args)

                bands_record.append(result.x.copy())

            bands[group_key] = tuple(bands_record)

        write_broad_optimum_records(self.scenario_name, evaluation_records, self.broad_optimum_var_info)
        write_broad_optimum_bands(
            self.scenario_name,
            self.broad_optimum_var_info,
            bands,
            self.get_differential_evolution_args(),
            band_lcoe_max,
            groups,
        )
        return bands

    def explore_midpoints(self) -> None:
        """
        Sample the near-optimal space by searching for solutions at evenly spaced
        midpoints between the min and max band endpoints for each group.

        Reads the previously computed band endpoints from file, divides each
        group's range into (midpoint_count + 1) equal steps, and runs a
        differential evolution optimisation targeting each interior step. Feasible
        solutions found at each step are appended to the midpoint CSV as they are
        discovered.

        Parameters:
        -------
        None.

        Returns:
        -------
        None.

        Side-effects:
        -------
        Reads near_optimal_bands.csv and appends feasible solution rows to
        midpoint_space.csv in the scenario's midpoint_explore results directory.
        Logs progress messages for each group and midpoint.

        Exceptions:
        -------
        None.
        """
        self.logger.info(f"[midpoint_explore] beginning midpoint exploration: {self.config.midpoint_count} per group")
        band_lcoe_max = self.get_band_lcoe_max()
        group_bands = read_broad_optimum_bands(self.scenario_name, self.broad_optimum_var_info)
        csv_path = create_midpoint_csv(self.scenario_name, self.broad_optimum_var_info)

        for group_key, bands in group_bands.items():
            band_max, band_min = float(bands["max"]), float(bands["min"])
            step_size = (band_max - band_min) / (self.config.midpoint_count + 1)
            idx_list = [
                variable["idx"] for variable in self.broad_optimum_var_info if (variable[0] or variable[3]) == group_key
            ]

            self.logger.info(
                f"[midpoint_explore] group '{group_key}'  min={band_min:.3f}  max={band_max:.3f}  step={step_size:.3f}"
            )

            for midpoint in range(1, self.config.midpoint_count + 1):
                evaluation_records = []
                group_target = band_min + midpoint * step_size
                self.logger.info(
                    f"[midpoint_explore] midpoint {midpoint}/{self.config.midpoint_count}: "
                    f"target sum ≈ {group_target:.3f}"
                )

                args = (
                    self.get_differential_evolution_args(),
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    "midpoint",
                    group_target,
                    midpoint,
                )

                self.run_differential_evolution(broad_optimum_objective, args)

                append_to_midpoint_csv(self.scenario_name, evaluation_records)

        self.logger.info(f"[midpoint_explore] finished; wrote {len(evaluation_records)} feasible points to {csv_path}")

        return None

    def capacity_expansion(self):
        return

    def evaluate(self) -> None:
        """
        Run the appropriate optimisation based on the config type.

        Parameters:
        -------
        None.

        Returns:
        -------
        None.

        Side-effects:
        -------
        Delegates to single_time(), find_near_optimal_band(), explore_midpoints(),
        or capacity_expansion() depending on self.config.type.

        Exceptions:
        -------
        Exception: Raised if self.config.type is not one of "single_time",
            "near_optimum", "midpoint_explore", or "capacity_expansion".
        """
        if self.config.type == "single_time":
            self.single_time()
        elif self.config.type == "near_optimum":
            self.find_near_optimal_band()
        elif self.config.type == "midpoint_explore":
            self.explore_midpoints()
        elif self.config.type == "capacity_expansion":
            self.capacity_expansion()
        else:
            raise Exception(
                "Model type in config must be 'single_time' or 'capacity_expansion' or 'near_optimum' or"
                "'midpoint_explore'"
            )


def callback(intermediate_result: OptimizeResult) -> None:
    """
    Differential evolution callback that saves iteration results to file.

    Called by scipy differential_evolution at the end of each iteration.
    Always appends the best solution vector to callback.csv. When SAVE_POPULATION
    is enabled, also appends the full population to population.csv, overwrites
    latest_population.csv with the current population, and appends population
    energies to population_energies.csv.

    Parameters:
    -------
    intermediate_result (OptimizeResult): Result object provided by scipy at the
        end of each iteration, containing at minimum the best solution vector (x)
        and optionally population and population_energies attributes.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Appends to or overwrites CSV files in the results/temp directory.

    Exceptions:
    -------
    None.
    """
    results_dir = os.path.join("results", "temp")
    os.makedirs(results_dir, exist_ok=True)

    # Save best solution from last iteration
    with open(os.path.join(results_dir, "callback.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(intermediate_result.x))

    if SAVE_POPULATION:
        # Save population from last iteration
        if hasattr(intermediate_result, "population"):
            with open(os.path.join(results_dir, "population.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                for individual in intermediate_result.population:
                    writer.writerow(list(individual))

            with open(os.path.join(results_dir, "latest_population.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                for individual in intermediate_result.population:
                    writer.writerow(list(individual))

        # Save population energies from last iteration
        if hasattr(intermediate_result, "population_energies"):
            with open(os.path.join(results_dir, "population_energies.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                for energy in intermediate_result.population_energies:
                    writer.writerow([energy])
