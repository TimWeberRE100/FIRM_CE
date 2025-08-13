import numpy as np
from numpy.typing import NDArray
from itertools import chain
from scipy.optimize import differential_evolution, OptimizeResult
import csv, os

from firm_ce.system.parameters import ModelConfig
from firm_ce.optimisation.single_time import evaluate_vectorised_xs
from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType

def fixed_path(root: str, scenario_name: str):
    base = os.path.join("results", root, scenario_name)
    os.makedirs(base, exist_ok=True)
    return base

class Solver:
    def __init__(self, 
                 config: ModelConfig, 
                 initial_x_candidate: NDArray[np.float64],
                 parameters_static: ScenarioParameters_InstanceType,
                 fleet_static: Fleet_InstanceType,
                 network_static: Network_InstanceType,
                 ) -> None:
        self.config = config
        self.decision_x0 = initial_x_candidate if len(initial_x_candidate) > 0 else None
        self.parameters_static = parameters_static
        self.fleet_static = fleet_static
        self.network_static = network_static
        self.lower_bounds, self.upper_bounds = self.get_bounds()
        #self._build_var_info()
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

    def get_de_args(self):
        args = (
            self.parameters_static,
            self.fleet_static,
            self.network_static,
            self.config.balancing_type,
        )
        return args

    def _single_time(self):
        self.initialise_callback()

        self.result = differential_evolution(
            x0=self.decision_x0,
            func=evaluate_vectorised_xs,
            bounds=list(zip(self.lower_bounds, self.upper_bounds)), 
            args = self.get_de_args(),
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
        
    """ 
    def _build_var_info(self):
        
        create a list of dicts mapping each decision variable index to:
          - its name
          - near_optimum on or off
          - its group key (to aggregate)
       
        self.var_info = []
        idx = 0
    
        for g in sorted(self.scenario.generators.values(), key=lambda g: g.id):
            if g.unit_type == 'solar':
                self.var_info.append({
                    'idx': idx,
                    'name': g.name,
                    'near_opt': g.near_opt,
                    'group':    g.group
                })
                idx += 1
                
        for g in sorted(self.scenario.generators.values(), key=lambda g: g.id):
            if g.unit_type == 'wind':
                self.var_info.append({
                    'idx':      idx,
                    'name':     g.name,
                    'near_opt': g.near_opt,
                    'group':    g.group
                })
                idx += 1
            
        for g in sorted(self.scenario.generators.values(), key=lambda g: g.id):
            if g.unit_type == 'flexible':
                self.var_info.append({
                    'idx':      idx,
                    'name':     g.name,
                    'near_opt': g.near_opt,
                    'group':    g.group
                })
                idx += 1
                
        for s in sorted(self.scenario.storages.values(), key=lambda s: s.id):
            idx_power = idx
            power_group = f"{s.group}_power" if s.group else idx_power
            self.var_info.append({
                'idx':      idx_power,
                'name':     f"{s.name}_power",
                'near_opt': s.near_opt,
                'group':    power_group
            })
            idx += 1
            
        for s in sorted(self.scenario.storages.values(), key=lambda s: s.id):
            idx_energy = idx
            energy_group = f"{s.group}_energy" if s.group else idx_energy
            self.var_info.append({
                'idx':      idx_energy,
                'name':     f"{s.name}_energy",
                'near_opt': s.near_opt,
                'group':    energy_group
            })
            idx += 1
            
        for l in sorted(self.scenario.lines.values(), key=lambda l: l.id):
            if not (is_nan(l.node_start) or is_nan(l.node_end)):
                self.var_info.append({
                    'idx':      idx,
                    'name':     f"{l.name}_line",
                    'near_opt': l.near_opt,      
                    'group':    l.group          
                })
                idx += 1

    def find_near_optimal_band(self):
    
        base_lcoe = self.optimal_lcoe
        tol       = self.config.near_optimal_tol
        band_max  = base_lcoe * (1 + tol)
        LARGE_PEN = 1e6
    
        items = [v for v in self.var_info if v['near_opt']]
        groups = {}
        for v in items:
            key = v['group'] or v['idx']
            groups.setdefault(key, []).append(v['idx'])
    
        all_evals = []
        bands     = {}
    
        de_args = self._construct_numba_classes()
    
        for group_key, idx_list in groups.items():
            
            self.scenario.logger.info(f"[near_optimum] exploring group '{group_key}'")
            
            def obj_batch_min(X): # 2-D array to allow vectorized DE
                batch = parallel_wrapper(X, *de_args)
                lcoes    = batch[1]
                pens     = batch[2]
                band_pen = np.maximum(0, lcoes - band_max) * LARGE_PEN
                var_sums = X[idx_list, :].sum(axis=0)
            
                for j in range(X.shape[1]):
                    if pens[j] <= 0.001 and band_pen[j] <= 0.001:
                        all_evals.append((
                            group_key,
                            'min',
                            float(lcoes[j]),
                            float(pens[j]),
                            float(band_pen[j]),
                            X[:, j].copy()
                        ))
                return band_pen + pens + var_sums
    
            def obj_batch_max(X):
                batch = parallel_wrapper(X, *de_args)
                lcoes    = batch[1]
                pens     = batch[2]
                band_pen = np.maximum(0, lcoes - band_max) * LARGE_PEN
                var_sums = X[idx_list, :].sum(axis=0)
                for j in range(X.shape[1]):
                    if pens[j] <= 0.001 and band_pen[j] <= 0.001:
                        all_evals.append((
                            group_key,
                            'max',
                            float(lcoes[j]),
                            float(pens[j]),
                            float(band_pen[j]),
                            X[:, j].copy()
                        ))
                return band_pen + pens - var_sums
    
            self.scenario.logger.info(f"[near_optimum] finding MIN for group '{group_key}'")
            res_min = differential_evolution(
                obj_batch_min,
                bounds=list(zip(self.lower_bounds, self.upper_bounds)),
                # args=args,  
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
                vectorized=True)
            
            self.scenario.logger.info(f"[near_optimum] finding MAX for group '{group_key}'")
            res_max = differential_evolution(
                obj_batch_max,
                bounds=list(zip(self.lower_bounds, self.upper_bounds)), 
                # args=args,  
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
                vectorized=True)
    
            bands[group_key] = (res_min.x.copy(), res_max.x.copy())
    
        space_dir  = fixed_path("near_optimum", self.scenario.name)
        space_path = os.path.join(space_dir, 'near_optimal_space.csv')
        with open(space_path, 'w', newline='') as f_space:
            writer_space = csv.writer(f_space)
            writer_space.writerow([
                'group', 'direction', 'lcoe', 'operational_penalty', 'band_penalty',
                *[f'x{i}' for i in range(len(self.lower_bounds))]
            ])
            for grp, direction, l, p, b, x_vec in all_evals:
                writer_space.writerow([grp, direction, l, p, b, *x_vec])
    
        bands_path = os.path.join(space_dir, 'near_optimal_bands.csv')
        near_vars = [v['name'] for v in self.var_info if v['near_opt']]
        name_to_col = {name:ci for ci,name in enumerate(near_vars)}
        with open(bands_path, 'w', newline='') as f_bands:
            writer_bands = csv.writer(f_bands)
            header = ['group', 'direction', 'lcoe', 'operational_penalty', 'band_penalty'] + near_vars
            writer_bands.writerow(header)
            
            for grp, (full_x_min, full_x_max) in bands.items():
                for direction,xvec in (('min',full_x_min),('max',full_x_max)):
                    sol = Solution_SingleTime(xvec, *args); sol.evaluate()
                    band_pen = max(0, sol.lcoe - band_max)*LARGE_PEN
                    row = [grp, direction, sol.lcoe, sol.penalties, band_pen]
                    vals = ['']*len(near_vars)
                    for idx in groups[grp]:
                        v = next(v for v in self.var_info if v['idx'] == idx)
                        col = name_to_col[v['name']]
                        vals[col] = xvec[idx]
                    writer_bands.writerow(row + vals)

        return bands
    
    def explore_midpoints(self, n_midpoints: int):
        
        self.scenario.logger.info(f"[midpoint_explore] beginning midpoint exploration: {n_midpoints} per group")
        
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
                
                self.scenario.logger.info(f"[midpoint_explore] group '{grp}'  min={mn:.3f}  max={mx:.3f}  step={step:.3f}")
                
                for i in range(1, n_midpoints+1):
                    target = mn + i * step
                    self.scenario.logger.info(f"[midpoint_explore] midpoint {i}/{n_midpoints}: target sum ≈ {target:.3f}")
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
                
        self.scenario.logger.info(f"[midpoint_explore] finished; wrote {len(all_mid_evals)} feasible points to {mid_csv}")
        
    def _capacity_expansion(self):
        pass

    def statistics(self, result_x):
        de_args = self._construct_numba_classes()
        solution = Solution_SingleTime(result_x, *de_args)
        solution.evaluate()

        return solution """

    def evaluate(self):
        if self.config.type not in ['single_time','capacity_expansion','near_optimum','midpoint_explore']:
            raise Exception("Model type in config must be 'single_time' or 'capacity_expansion' or 'near_optimum' or 'midpoint_explore'")

        if self.config.type == 'single_time':
            self._single_time()
            
        """ elif self.config.type == 'near_optimum':
            self.optimal_lcoe = self.config.global_optimal_lcoe
            self.find_near_optimal_band()
            
        elif self.config.type == 'midpoint_explore':
            self.optimal_lcoe = self.config.global_optimal_lcoe
            self.explore_midpoints(self.config.midpoint_count)
            
        elif self.config.type == 'capacity_expansion':
            self._capacity_expansion()  """

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