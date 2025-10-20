import csv
import os
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import time

import numpy as np

from firm_ce.common.constants import PENALTY_MULTIPLIER, TOLERANCE, NUM_THREADS
from firm_ce.common.typing import (
    BandCandidates_Type,
    BroadOptimumVars_Type,
    EvaluationRecord_Type,
)
from firm_ce.optimisation.single_time import Solution, parallel_wrapper
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType

class BroadOptimum:
    @staticmethod
    def get_results_path(results_root: str, scenario_name: str) -> str:
        results_path = os.path.join("results", results_root, scenario_name)
        os.makedirs(results_path, exist_ok=True)
        return results_path
    
    @staticmethod
    def create_variable_record(
        candidate_x_idx: int, 
        asset_name: str, 
        units: str, 
        near_optimum_check: bool, 
        near_optimum_group: str,
        min_build: float,
        max_build: float
    ) -> BroadOptimumVars_Type:
        return (
            candidate_x_idx,
            asset_name,
            units,
            near_optimum_check,
            near_optimum_group,
            min_build,
            max_build
        )
    
    @staticmethod
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
        return (
            group_key,
            band_type,
            target_group_var_sum if target_group_var_sum else "N/A",
            float(population_lcoes[solution_index]),
            float(de_population_penalties[solution_index]),
            float(band_population_penalties[solution_index]),
            band_population_candidates[:, solution_index].copy(),
        )
    
    @staticmethod
    def get_type(model_type: str) -> str | None:
        broad_optimum_type = None
        if "min" in model_type:
            broad_optimum_type = "min"
        elif "max" in model_type:
            broad_optimum_type = "max"
        return broad_optimum_type
        
    def __init__(self, 
                 scenario_name: str, 
                 model_type: str, 
                 fleet: Fleet_InstanceType, 
                 network: Network_InstanceType,
                 lcoe_cutoff: float
                 ):
        self.type = self.get_type(model_type)
        self.results_path = self.get_results_path("near_optimum", scenario_name)        
        self.variable_info = self.build_variable_info(fleet, network)
        self.evaluation_records = []
        
        self.lcoe_cutoff = lcoe_cutoff
        self.bands = {}
        self.groups = self.build_groups_dict()

    def build_groups_dict(self):
        groups = {}
        for record in self.variable_info:
            candidate_x_idx, _, _, near_optimum_check, group, _, _ = record

            if not near_optimum_check:
                continue
            key = group or candidate_x_idx
            groups.setdefault(key, []).append(candidate_x_idx)
        return groups

    def build_variable_info(
        self, fleet: Fleet_InstanceType, network: Network_InstanceType
    ) -> List[BroadOptimumVars_Type]:
        variable_info = []

        for generator in fleet.generators.values():
            variable_info.append(
                self.create_variable_record(
                    generator.candidate_x_idx, 
                    generator.name, 
                    "GW", 
                    generator.near_optimum_check, 
                    generator.group, 
                    generator.min_build, 
                    generator.max_build
                )
            )

        for storage in fleet.storages.values():
            variable_info.append(
                self.create_variable_record(
                    storage.candidate_p_x_idx, 
                    storage.name, 
                    "GW", 
                    storage.near_optimum_check, 
                    storage.group,
                    storage.min_build_p,
                    storage.max_build_p
                )
            )

        for storage in fleet.storages.values():
            variable_info.append(
                self.create_variable_record(
                    storage.candidate_e_x_idx, 
                    storage.name, 
                    "GWh", 
                    storage.near_optimum_check, 
                    storage.group,
                    storage.min_build_e,
                    storage.max_build_e
                )
            )

        for line in network.major_lines.values():
            variable_info.append(
                self.create_variable_record(
                    line.candidate_x_idx, 
                    line.name, 
                    "GW", 
                    line.near_optimum_check, 
                    line.group,
                    line.min_build,
                    line.max_build
                )
            )
        return variable_info
    
    def add_band_record(self, group_key: str, band_x: NDArray[np.float64]) -> None:
        self.bands[group_key] = band_x.copy()
        return None

    def write_records(self) -> None:
        space_path = os.path.join(self.results_path, "near_optimal_space.csv")
        with open(space_path, "w+", newline="") as records_file:
            writer_space = csv.writer(records_file)
            writer_space.writerow(
                [
                    "Group",
                    "Band_Type",
                    "LCOE [$/MWh]",
                    "Operational_Penalty",
                    "Band_Penalty",
                    *[f"{asset_name} [{unit}]" for _, asset_name, unit, _, _, _, _ in self.variable_info],
                ]
            )
            for group, band_type, _, lcoe, unit_commitment_penalty, band_penalty, candidate_x in self.evaluation_records:
                writer_space.writerow([group, band_type, lcoe, unit_commitment_penalty, band_penalty, *candidate_x])
        return None

    def write_bands(self, unit_commitment_args: Tuple) -> None:   
        bands_path = os.path.join(self.results_path, "near_optimal_bands.csv")
        asset_column_names = [
            f"{asset_name} [{unit}]" for _, asset_name, unit, near_optimum_check, _, _, _ in self.variable_info if near_optimum_check
        ]
        names_to_columns = {name: col for col, name in enumerate(asset_column_names)}

        with open(bands_path, "w+", newline="") as bands_file:
            writer_bands = csv.writer(bands_file)
            writer_bands.writerow(
                [
                    "Group",
                    "Band_Type",
                    "LCOE [$/MWh]",
                    "Operational_Penalty",
                    "Band_Penalty",
                    *asset_column_names
                ]
            )

            for group, candidate_x in self.bands.items():
                solution = Solution(candidate_x, *unit_commitment_args)
                solution.evaluate()
                band_penalty = max(0, solution.lcoe - self.lcoe_cutoff) * PENALTY_MULTIPLIER
                group_values = [group, self.type, solution.lcoe, solution.penalties, band_penalty]
                asset_values = [""] * len(asset_column_names)
                for candidate_x_idx in self.groups[group]:
                    _, asset_name, unit, _, _, _, _ = self.variable_info[candidate_x_idx]
                    col = names_to_columns[f"{asset_name} [{unit}]"]
                    asset_values[col] = candidate_x[candidate_x_idx]
                writer_bands.writerow(group_values + asset_values)
        return None

    def minimise_group_xs(self, initial_population: NDArray[np.float64], group_key: str) -> NDArray[np.float64]:
        initial_population_adjusted = initial_population.copy()
        for candidate_x_idx, _, _, near_optimum_check, near_optimum_group, min_build, _ in self.variable_info:
            if near_optimum_check and (near_optimum_group == group_key):
                initial_population_adjusted[:, candidate_x_idx] = min_build
        return initial_population_adjusted


    def maximise_group_xs(self, initial_population: NDArray[np.float64], group_key: str) -> NDArray[np.float64]:
        initial_population_adjusted = initial_population.copy()
        for row in range(initial_population.shape[0]):
            for candidate_x_idx, _, _, near_optimum_check, near_optimum_group, _, max_build in self.variable_info:
                if near_optimum_check and (near_optimum_group == group_key):
                    initial_population_adjusted[row, candidate_x_idx] = max_build
        return initial_population_adjusted


def broad_optimum_objective(
    band_population_candidates: NDArray[np.float64],  # 2-dimensional array to allow vectorised DE
    unit_commitment_args,
    broad_optimum: BroadOptimum,
    group_key: str,
    group_orders: List[int]
) -> float:
    start_time = time.time()
    _, population_lcoes, population_penalties = parallel_wrapper(
        band_population_candidates, *unit_commitment_args
    )
    end_time = time.time()
    print(f"Average objective time: {NUM_THREADS*(end_time-start_time)/band_population_candidates.shape[1]:.4f} seconds.")
    print(f"Iteration time: {(end_time-start_time):.4f} seconds for {NUM_THREADS} workers.")
    
    group_variable_sums = band_population_candidates[group_orders, :].sum(axis=0)
    band_population_penalties = np.maximum(0, population_lcoes - broad_optimum.lcoe_cutoff) * PENALTY_MULTIPLIER

    for candidate_x in range(band_population_candidates.shape[1]):
        if not (
            population_penalties[candidate_x] <= TOLERANCE and band_population_penalties[candidate_x] <= TOLERANCE
        ):
            continue
        broad_optimum.evaluation_records.append(
            broad_optimum.create_evaluation_record(
                group_key,
                broad_optimum.type,
                population_lcoes,
                population_penalties,
                band_population_penalties,
                band_population_candidates,
                candidate_x,
            )
        )
    update_records_time = time.time()
    print(f"Time to update broad optimum records: {(update_records_time-end_time):.4f} seconds.")

    match broad_optimum.type:
        case "min":
            return band_population_penalties + population_penalties + group_variable_sums
        case "max":
            return band_population_penalties + population_penalties - group_variable_sums
        case _:
            return None
