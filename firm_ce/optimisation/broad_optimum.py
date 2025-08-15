from typing import List, Dict
import numpy as np
import os, csv

from firm_ce.common.constants import PENALTY_MULTIPLIER
from firm_ce.optimisation.single_time import parallel_wrapper
from firm_ce.common.typing import (
    EvaluationRecord_Type, 
    DifferentialEvolutionArgs_Type, 
    BroadOptimumVars_Type,
    BandCandidates_Type,
)
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.optimisation.single_time import Solution

def near_optimum_path(root: str, scenario_name: str):
    base = os.path.join("results", root, scenario_name)
    os.makedirs(base, exist_ok=True)
    return base

def create_broad_optimum_vars_record(candidate_x_idx: int,
                                     asset_name: str,
                                     near_optimum_check: bool,
                                     near_optimum_group: str
                                     ) -> BroadOptimumVars_Type:
    return (
        candidate_x_idx,
        asset_name,
        near_optimum_check,
        near_optimum_group,
    )

def build_broad_optimum_var_info(fleet: Fleet_InstanceType, network: Network_InstanceType) -> List[BroadOptimumVars_Type]:        
    """ create a list of records mapping each decision variable index to:
        - its name
        - near_optimum on or off
        - its group key (to aggregate) """
    
    broad_optimum_var_info = []

    for generator in fleet.generators.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                generator.candidate_x_idx,
                generator.name,
                generator.near_optimum_check,
                generator.group
            )                
        )
            
    for storage in fleet.storages.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_p_x_idx,
                storage.name,
                storage.near_optimum_check,
                storage.group
            )                
        )
        
    for storage in fleet.storages.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_e_x_idx,
                storage.name,
                storage.near_optimum_check,
                storage.group
            )                
        )
        
    for line in network.major_lines.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                line.candidate_x_idx,
                line.name,
                line.near_optimum_check,
                line.group
            )                
        )
    return broad_optimum_var_info

def create_evaluation_record(group_key:  str,
                             band_type:  str,
                             population_lcoes: List[float],
                             de_population_penalties: List[float],
                             band_population_penalties: List[float],
                             band_population_candidates: List[List[float]],
                             solution_index: int,
                             ) -> EvaluationRecord_Type:
    return (
        group_key,
        band_type,
        float(population_lcoes[solution_index]),
        float(de_population_penalties[solution_index]),
        float(band_population_penalties[solution_index]),
        band_population_candidates[:, solution_index].copy()
    )

def broad_optimum_objective(band_population_candidates: List[List[float]], # 2-D array to allow vectorized DE
                            differential_evolution_args: DifferentialEvolutionArgs_Type,
                            group_key: str,
                            band_lcoe_max: float,
                            bo_group_orders: List[int],
                            evaluation_records: List[EvaluationRecord_Type],
                            band_type: str,
                            ) -> float: 
    
    _, population_lcoes, de_population_penalties = parallel_wrapper(band_population_candidates, *differential_evolution_args)
    group_var_sums = band_population_candidates[bo_group_orders, :].sum(axis=0)
    band_population_penalties = np.maximum(0, population_lcoes - band_lcoe_max) * PENALTY_MULTIPLIER

    for candidate_x in range(band_population_candidates.shape[1]):
        """ if not (de_population_penalties[candidate_x] <= 0.001 and band_population_penalties[candidate_x] <= 0.001):
            continue """ #### DEBUG ########
        evaluation_records.append(
            create_evaluation_record(
                group_key,
                band_type,
                population_lcoes,
                de_population_penalties,
                band_population_penalties,
                band_population_candidates,
                candidate_x
            )
        )
    
    match band_type:
        case 'min':
            return band_population_penalties + de_population_penalties + group_var_sums
        case 'max':
            return band_population_penalties + de_population_penalties - group_var_sums
        case _:
            return None   
        
def write_broad_optimum_records(scenario_name: str,
                           evaluation_records: List[EvaluationRecord_Type],
                           broad_optimum_var_info: List[BroadOptimumVars_Type]) -> None:
    space_dir  = near_optimum_path("near_optimum", scenario_name)

    space_path = os.path.join(space_dir, 'near_optimal_space.csv')
    with open(space_path, 'w', newline='') as f_space:
        writer_space = csv.writer(f_space)
        writer_space.writerow([
            'Group', 'Band_Type', 'LCOE [$/MWh]', 'Operational_Penalty', 'Band_Penalty',
            *[f'{asset_name}' for _, asset_name, _, _ in broad_optimum_var_info]
        ])
        for group, band_type, lcoe, de_penalty, band_penalty, candidate_x in evaluation_records:
            writer_space.writerow([
                group, 
                band_type, 
                lcoe, 
                de_penalty, 
                band_penalty, 
                *candidate_x
            ])
    return None

def write_broad_optimum_bands(scenario_name: str,
                           broad_optimum_var_info: List[BroadOptimumVars_Type],
                           bands: BandCandidates_Type,
                           de_args: DifferentialEvolutionArgs_Type,
                           band_lcoe_max: float,
                           groups: Dict[str, List[int]]) -> None:
    space_dir  = near_optimum_path("near_optimum", scenario_name)
    bands_path = os.path.join(space_dir, 'near_optimal_bands.csv')

    near_optimal_asset_names = [asset_name for _, asset_name, near_optimum_check, _ in broad_optimum_var_info if near_optimum_check]
    names_to_columns = {name: col for col, name in enumerate(near_optimal_asset_names)}

    with open(bands_path, 'w', newline='') as f_bands:
        writer_bands = csv.writer(f_bands)
        header = ['Group', 'Band_Type', 'LCOE [$/MWh]', 'Operational_Penalty', 'Band_Penalty'] + near_optimal_asset_names
        writer_bands.writerow(header)
        
        for group, (candidate_x_min, candidate_x_max) in bands.items():
            for band_type, candidate_x in (('min',candidate_x_min),('max',candidate_x_max)):
                solution = Solution(candidate_x, *de_args); solution.evaluate()
                band_penalty = max(0, solution.lcoe - band_lcoe_max)*PENALTY_MULTIPLIER
                row = [group, band_type, solution.lcoe, solution.penalties, band_penalty]
                vals = ['']*len(near_optimal_asset_names)
                for candidate_x_idx in groups[group]:
                    _, asset_name, _, _ = broad_optimum_var_info[candidate_x_idx]
                    col = names_to_columns[asset_name]
                    vals[col] = candidate_x[candidate_x_idx]
                writer_bands.writerow(row + vals)
    return None