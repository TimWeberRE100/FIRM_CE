import numpy as np
from numpy.typing import NDArray
from typing import List
from itertools import chain
from scipy.optimize import differential_evolution, OptimizeResult
import csv, os
from logging import Logger

from firm_ce.system.parameters import ModelConfig
from firm_ce.optimisation.single_time import evaluate_vectorised_xs
from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.optimisation.broad_optimum import (
    build_broad_optimum_var_info, 
    broad_optimum_objective,
    write_broad_optimum_records,
    write_broad_optimum_bands,
)
from firm_ce.common.typing import (
    EvaluationRecord_Type, 
    DifferentialEvolutionArgs_Type, 
    BroadOptimumVars_Type
)

class Solver:
    def __init__(self, 
                 config: ModelConfig, 
                 initial_x_candidate: NDArray[np.float64],
                 parameters_static: ScenarioParameters_InstanceType,
                 fleet_static: Fleet_InstanceType,
                 network_static: Network_InstanceType,
                 scenario_logger: Logger,
                 scenario_name: str,
                 ) -> None:
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

    def get_bounds(self):
        def power_capacity_bounds(asset_list, build_cap_constraint):
            return [
                getattr(asset, build_cap_constraint)
                for asset in asset_list
            ]

        def energy_capacity_bounds(storage_list, build_cap_constraint):
            return [
                getattr(s, build_cap_constraint)
                if s.duration == 0
                else 0.0
                for s in storage_list
            ]
        
        generators = list(self.fleet_static.generators.values())
        storages = list(self.fleet_static.storages.values())
        lines = list(self.network_static.major_lines.values())

        lower_bounds = np.array(list(chain(
            power_capacity_bounds(generators, "min_build"),
            power_capacity_bounds(storages, "min_build_p"),
            energy_capacity_bounds(storages, "min_build_e"),
            power_capacity_bounds(lines, "min_build"),
        )))

        upper_bounds = np.array(list(chain(
            power_capacity_bounds(generators, "max_build"),
            power_capacity_bounds(storages, "max_build_p"),
            energy_capacity_bounds(storages, "max_build_e"),
            power_capacity_bounds(lines, "max_build"),
        )))

        return lower_bounds, upper_bounds

    def initialise_callback(self):
        temp_dir = os.path.join("results", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "callback.csv"), 'w', newline='') as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population.csv"), 'w', newline='') as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population_energies.csv"), 'w', newline='') as csvfile:
            csv.writer(csvfile)

    def get_differential_evolution_args(self):
        args = (
            self.parameters_static,
            self.fleet_static,
            self.network_static,
            self.config.balancing_type,
        )
        return args

    def single_time(self):
        self.initialise_callback()

        self.result = differential_evolution(
            x0=self.decision_x0,
            func=evaluate_vectorised_xs,
            bounds=list(zip(self.lower_bounds, self.upper_bounds)), 
            args = self.get_differential_evolution_args(),
            tol=0,
            maxiter=self.config.iterations, 
            popsize=self.config.population, 
            mutation=(0.2,self.config.mutation), 
            recombination=self.config.recombination,
            disp=True, 
            polish=False, 
            updating='deferred',
            callback=callback, 
            workers=1,
            vectorized=True,
        )

    def find_near_optimal_band(self):    
        band_lcoe_max = self.optimal_lcoe * (1 + self.config.near_optimal_tol)
    
        groups = {}
        for record in self.broad_optimum_var_info:
            candidate_x_idx, _, near_optimum_check, group = record

            if not near_optimum_check:
                continue
            key = group or candidate_x_idx
            groups.setdefault(key, []).append(candidate_x_idx)
    
        evaluation_records = []
        bands = {}
    
        for group_key, idx_list in groups.items():            
            self.logger.info(f"[near_optimum] exploring group '{group_key}'")

            bands_record = []
            for band_type in ('min', 'max'):
                match band_type:
                    case 'min':
                        self.logger.info(f"[near_optimum] finding MIN for group '{group_key}'")
                    case 'max':
                        self.logger.info(f"[near_optimum] finding MAX for group '{group_key}'")

                args = (
                    self.get_differential_evolution_args(), 
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    band_type
                )
            
                result = differential_evolution(
                    broad_optimum_objective,
                    bounds=list(zip(self.lower_bounds, self.upper_bounds)),
                    args=args,  
                    tol=0,
                    maxiter=self.config.iterations, 
                    popsize=self.config.population, 
                    mutation=(0.2,self.config.mutation), 
                    recombination=self.config.recombination,
                    disp=True, 
                    polish=False, 
                    updating='deferred',
                    callback=callback, 
                    workers=1,
                    vectorized=True
                )

                bands_record.append(result.x.copy())
            
            bands[group_key] = tuple(bands_record)

        write_broad_optimum_records(self.scenario_name, evaluation_records, self.broad_optimum_var_info)
        write_broad_optimum_bands(self.scenario_name, 
                                  self.broad_optimum_var_info, 
                                  bands, 
                                  self.get_differential_evolution_args(),
                                  band_lcoe_max,
                                  groups
                                  )
        return bands
    
    """ def explore_midpoints(self, n_midpoints: int):
        
        self.logger.info(f"[midpoint_explore] beginning midpoint exploration: {n_midpoints} per group")
        
        base_lcoe = self.optimal_lcoe
        tol       = self.config.near_optimal_tol
        band_max  = base_lcoe * (1 + tol)
        LARGE_PEN = 1e6
        
        name_groups: Dict[str,List[str]] = {}
        for v in self.var_info:
            if v['near_opt']:
                name_groups.setdefault(v['group'], []).append(v['name'])
        
        bands_dir = fixed_path("near_optimum", self.scenario.name)
        bands_path = os.path.join(bands_dir, "near_optimal_bands.csv")
        groups: Dict[str, Dict[str, float]] = {}
        
        with open(bands_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                grp = row['group']
                direction = row['direction']
                vals = [float(row[var_name]) for var_name in name_groups.get(grp, [])]
                s = sum(vals)
                groups.setdefault(grp, {'min': None, 'max': None})[direction] = s
                
        de_args = self._construct_numba_classes()
        
        mid_dir = fixed_path("midpoint_explore", self.scenario.name)
        mid_csv = os.path.join(mid_dir, "midpoint_space.csv")
        all_mid_evals = []
        with open(mid_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'group', 'midpoint_idx', 'target',
                'lcoe', 'operational_penalty', 'band_penalty',
                *[f'x{i}' for i in range(len(self.lower_bounds))]
            ])
            for grp, limits in groups.items():
                mn, mx = limits['min'], limits['max']
                step = (mx - mn) / (n_midpoints + 1)
                idx_list = [v['idx'] for v in self.var_info if (v['group'] or v['idx']) == grp]
                
                self.logger.info(f"[midpoint_explore] group '{grp}'  min={mn:.3f}  max={mx:.3f}  step={step:.3f}")
                
                for i in range(1, n_midpoints+1):
                    target = mn + i * step
                    self.logger.info(f"[midpoint_explore] midpoint {i}/{n_midpoints}: target sum ≈ {target:.3f}")
                    def obj_mid(X):
                        batch = parallel_wrapper(X, *de_args)
                        lcoes = batch[1]
                        pens = batch[2]
                        band_pen = np.maximum(0, lcoes - band_max) * LARGE_PEN
                        var_sum  = X[idx_list, :].sum(axis=0)
                        target_pen = np.abs(var_sum - target)
                        
                        for j in range(X.shape[1]):
                            if pens[j] <= 1e-3 and band_pen[j] <= 1e-3:
                                all_mid_evals.append([
                                    grp,
                                    i,
                                    target,
                                    float(lcoes[j]),
                                    float(pens[j]),
                                    float(band_pen[j]),
                                    *X[:, j].tolist()
                                ])
                        return band_pen + pens + target_pen
                    
                    differential_evolution(
                        obj_mid,
                        bounds=list(zip(self.lower_bounds, self.upper_bounds)),
                        tol=0,
                        maxiter=self.config.iterations,
                        popsize=self.config.population,
                        mutation=(0.2,self.config.mutation),
                        recombination=self.config.recombination,
                        disp=True,
                        polish=False,
                        updating='deferred',
                        callback=callback,
                        workers=1,
                        vectorized=True
                    )
                    
            for row in all_mid_evals:
                writer.writerow(row)
                
        self.logger.info(f"[midpoint_explore] finished; wrote {len(all_mid_evals)} feasible points to {mid_csv}")
         """
    def capacity_expansion(self):
        pass

    def evaluate(self):
        if self.config.type == 'single_time':
            self.single_time()            
        elif self.config.type == 'near_optimum':
            self.optimal_lcoe = self.config.global_optimal_lcoe
            self.find_near_optimal_band()            
            """ elif self.config.type == 'midpoint_explore':
                self.optimal_lcoe = self.config.global_optimal_lcoe
                self.explore_midpoints(self.config.midpoint_count)  """           
        elif self.config.type == 'capacity_expansion':
            self.capacity_expansion() 
        else:
            raise Exception("Model type in config must be 'single_time' or 'capacity_expansion' or 'near_optimum' or 'midpoint_explore'")

def callback(intermediate_result: OptimizeResult) -> None:
    results_dir = os.path.join("results", "temp")
    os.makedirs(results_dir, exist_ok=True)

    # Save best solution from last iteration
    with open(os.path.join(results_dir, "callback.csv"), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(intermediate_result.x))

    if SAVE_POPULATION:
        # Save population from last iteration
        if hasattr(intermediate_result, "population"):
            with open(os.path.join(results_dir, "population.csv"), 'a', newline='') as f:
                writer = csv.writer(f)
                for individual in intermediate_result.population:
                    writer.writerow(list(individual))

        # Save population energies from last iteration
        if hasattr(intermediate_result, "population_energies"):
            with open(os.path.join(results_dir, "population_energies.csv"), 'a', newline='') as f:
                writer = csv.writer(f)
                for energy in intermediate_result.population_energies:
                    writer.writerow([energy])