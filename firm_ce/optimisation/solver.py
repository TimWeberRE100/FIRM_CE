import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from firm_ce.optimisation.single_time import parallel_wrapper, Solution_SingleTime, parallel_wrapper_lp
import csv, os

from firm_ce.io.validate import is_nan
from firm_ce.common.constants import SAVE_POPULATION
from typing import Dict, List

def fixed_path(root: str, scenario_name: str):
    base = os.path.join("results", root, scenario_name)
    os.makedirs(base, exist_ok=True)
    return base

class Solver:
    def __init__(self, config, scenario) -> None:
        self.config = config
        self.scenario = scenario
        self.decision_x0 = scenario.x0 if len(scenario.x0) > 0 else None
        self.lower_bounds, self.upper_bounds = self._get_bounds()
        self._build_var_info()
        self.result = None
        
    def _build_var_info(self):
        """
        create a list of dicts mapping each decision variable index to:
          - its name
          - near_optimum on or off
          - its group key (to aggregate)
        """
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

    def _get_bounds(self):
        solar_generators = [self.scenario.generators[idx] for idx in self.scenario.generators if self.scenario.generators[idx].unit_type == 'solar']
        wind_generators = [self.scenario.generators[idx] for idx in self.scenario.generators if self.scenario.generators[idx].unit_type == 'wind']
        flexible_generators = [self.scenario.generators[idx] for idx in self.scenario.generators if self.scenario.generators[idx].unit_type == 'flexible']
        storages = [self.scenario.storages[idx] for idx in self.scenario.storages]
        lines = [self.scenario.lines[idx] for idx in self.scenario.lines]

        solar_lb = [generator.capacity + generator.min_build for generator in solar_generators]
        wind_lb = [generator.capacity + generator.min_build for generator in wind_generators]
        flexible_p_lb = [generator.capacity + generator.min_build for generator in flexible_generators]
        storage_p_lb = [storage.power_capacity + storage.min_build_p for storage in storages] 
        storage_e_lb = [storage.energy_capacity + storage.min_build_e if storage.duration == 0 else 0.0 for storage in storages] 
        line_lb = [line.capacity + line.min_build for line in lines if not (is_nan(line.node_start) or is_nan(line.node_end))]
        lower_bounds = np.array(solar_lb + wind_lb + flexible_p_lb + storage_p_lb + storage_e_lb + line_lb)

        solar_ub = [generator.capacity + generator.max_build for generator in solar_generators]
        wind_ub = [generator.capacity + generator.max_build for generator in wind_generators]
        flexible_p_ub = [generator.capacity + generator.max_build for generator in flexible_generators] 
        storage_p_ub = [storage.power_capacity + storage.max_build_p for storage in storages] 
        storage_e_ub = [storage.energy_capacity + storage.max_build_e if storage.duration == 0 else 0.0 for storage in storages]
        line_ub = [line.capacity + line.max_build for line in lines if not (is_nan(line.node_start) or is_nan(line.node_end))]
        upper_bounds = np.array(solar_ub + wind_ub + flexible_p_ub + storage_p_ub + storage_e_ub + line_ub)

        return lower_bounds, upper_bounds

    def _prepare_scenario_arrays(self):
        
        scenario_arrays = {}
        node_names = {self.scenario.nodes[idx].name : self.scenario.nodes[idx].id for idx in self.scenario.nodes}

        # Static parameters
        scenario_arrays['MLoad'] = np.array(
            [self.scenario.nodes[idx].demand_data
            for idx in range(0, max(self.scenario.nodes)+1)],
            dtype=np.float64, ndmin=2
        ).T

        scenario_arrays['intervals'], scenario_arrays['nodes'] = scenario_arrays['MLoad'].shape
        scenario_arrays['lines'] = len(self.scenario.lines)
        scenario_arrays['years'] = self.scenario.final_year - self.scenario.first_year + 1
        scenario_arrays['resolution'] = self.scenario.resolution
        scenario_arrays['allowance'] = self.scenario.allowance

        # Unit types
        scenario_arrays['generator_unit_types_setting'] = {self.config.settings.generator_unit_types[idx]['unit_type'] : idx for idx in self.config.settings.generator_unit_types}
        scenario_arrays['storage_unit_types_setting'] = {self.config.settings.storage_unit_types[idx]['unit_type'] : idx for idx in self.config.settings.storage_unit_types}

        # Generators
        scenario_arrays['generator_ids'] = np.array(
            [self.scenario.generators[idx].id 
            for idx in self.scenario.generators],
            dtype=np.int64
        )

        scenario_arrays['generator_line_ids'] = np.array(
            [self.scenario.generators[idx].line.id 
            for idx in self.scenario.generators],
            dtype=np.int64
        )

        scenario_arrays['generator_nodes'] = np.array(
            [node_names[self.scenario.generators[idx].node]
            for idx in self.scenario.generators],
            dtype=np.int64
        )

        scenario_arrays['generator_capacities'] = np.array(
            [self.scenario.generators[idx].capacity 
            for idx in self.scenario.generators],
            dtype=np.float64
        )

        scenario_arrays['generator_unit_types'] = np.array(
            [scenario_arrays['generator_unit_types_setting'][self.scenario.generators[idx].unit_type]
            for idx in self.scenario.generators],
            dtype=np.int64
        )

        scenario_arrays['generator_unit_size'] = np.zeros(
            (max(self.scenario.generators)+1), dtype=np.float64
        )

        scenario_arrays['generator_costs'] = np.zeros(
            (8, max(self.scenario.generators)+1), dtype=np.float64
        )
        
        scenario_arrays['TSPV'] = np.array([
            self.scenario.generators[idx].data
            for idx in self.scenario.generators
            if self.scenario.generators[idx].unit_type == 'solar'
        ], dtype=np.float64, ndmin=2).T

        scenario_arrays['TSWind'] = np.array([
            self.scenario.generators[idx].data
            for idx in self.scenario.generators
            if self.scenario.generators[idx].unit_type == 'wind'
        ], dtype=np.float64, ndmin=2).T

        scenario_arrays['TSBaseload'] = np.array([
            self.scenario.generators[idx].data
            for idx in self.scenario.generators
            if self.scenario.generators[idx].unit_type == 'baseload'
        ], dtype=np.float64, ndmin=2).T

        scenario_arrays['Flexible_Limits_Annual'] = np.array([
            self.scenario.generators[idx].annual_limit
            for idx in self.scenario.generators
            if self.scenario.generators[idx].unit_type == "flexible"
        ], dtype=np.float64, ndmin=2).T

        scenario_arrays['solar_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['solar'])[0]] 
        scenario_arrays['wind_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['wind'])[0]] 
        scenario_arrays['flexible_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['flexible'])[0]] 
        scenario_arrays['baseload_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['baseload'])[0]] 

        # Storages
        scenario_arrays['storage_ids'] = np.array(
            [self.scenario.storages[idx].id 
            for idx in self.scenario.storages],
            dtype=np.int64
        )

        scenario_arrays['storage_line_ids'] = np.array(
            [self.scenario.storages[idx].line.id 
            for idx in self.scenario.storages],
            dtype=np.int64
        )

        scenario_arrays['storage_nodes'] = np.array(
                                                [node_names[self.scenario.storages[idx].node]
                                                for idx in self.scenario.storages],
                                                dtype=np.int64
                                            )

        scenario_arrays['storage_durations'] = np.array(
            [self.scenario.storages[idx].duration
            for idx in self.scenario.storages],
            dtype=np.float64
        )

        scenario_arrays["storage_d_efficiencies"] = np.array(
            [self.scenario.storages[idx].discharge_efficiency
            for idx in self.scenario.storages],
            dtype=np.float64
        )

        scenario_arrays["storage_c_efficiencies"] = np.array(
            [self.scenario.storages[idx].charge_efficiency
            for idx in self.scenario.storages],
            dtype=np.float64
        )

        scenario_arrays['storage_costs'] = np.zeros(
            (7, max(self.scenario.storages)+1), dtype=np.float64
        )

        scenario_arrays['Discharge'] = np.zeros(
            (scenario_arrays['intervals'], len(self.scenario.storages)), dtype=np.float64
        )
        scenario_arrays['Charge'] = np.zeros(
            (scenario_arrays['intervals'], len(self.scenario.storages)), dtype=np.float64
        )

        # Flexible
        scenario_arrays['flexible_ids'] = np.array(
                                            [self.scenario.generators[idx].id 
                                            for idx in self.scenario.generators
                                            if self.scenario.generators[idx].unit_type == "flexible"],
                                            dtype=np.int64
                                        )

        # Lines
        scenario_arrays['line_ids'] = np.array(
            [self.scenario.lines[idx].id for idx in self.scenario.lines],
            dtype=np.int64
        )

        minor_line_ids = np.array(
            [self.scenario.lines[idx].id for idx in self.scenario.lines
            if (is_nan(self.scenario.lines[idx].node_start) or is_nan(self.scenario.lines[idx].node_end))],
            dtype=np.int64
        )

        scenario_arrays['line_lengths'] = np.array(
            [self.scenario.lines[idx].length for idx in self.scenario.lines],
            dtype=np.float64
        )

        scenario_arrays['line_costs'] = np.zeros(
            (7, max(self.scenario.lines)+1), dtype=np.float64
        )

        scenario_arrays['network'] = self.scenario.network.network # ndmin?
        scenario_arrays['networksteps'] = self.scenario.network.networksteps # ndmin?
        scenario_arrays['transmission_mask'] = self.scenario.network.transmission_mask # ndmin?

        scenario_arrays['TLoss'] = np.array(
            [self.scenario.lines[idx].loss_factor for idx in self.scenario.lines if (not is_nan(self.scenario.lines[idx].node_start)) and (not is_nan(self.scenario.lines[idx].node_end))],
            dtype=np.float64
        )

        scenario_arrays['TFlows'] = np.zeros(
            (scenario_arrays['intervals'], len(self.scenario.lines)), dtype=np.float64
        )
        scenario_arrays['TFlowsAbs'] = np.zeros(
            (scenario_arrays['intervals'], len(self.scenario.lines)), dtype=np.float64
        )       

        # Decision variable indices
        scenario_arrays['pv_idx'] = scenario_arrays['TSPV'].shape[1]
        scenario_arrays['wind_idx'] = scenario_arrays['pv_idx'] + scenario_arrays['TSWind'].shape[1]
        scenario_arrays['flexible_p_idx'] = scenario_arrays['wind_idx'] + len(scenario_arrays['flexible_ids'])
        scenario_arrays['storage_p_idx'] = scenario_arrays['flexible_p_idx'] + len(self.scenario.storages) 
        scenario_arrays['storage_e_idx'] = scenario_arrays['storage_p_idx'] + len(self.scenario.storages) 
        scenario_arrays['lines_idx'] = scenario_arrays['storage_e_idx'] + len(self.scenario.lines)

        # Costs
        '''
        np array of form:
            (
            [capex_p]
            [capex_e]
            [fom]
            [vom]
            [lifetime]
            [discount_rate]
            [transformer_capex]
            [lcoe]
            )
        '''       
        
        for i, idx in enumerate(self.scenario.generators):
            g = self.scenario.generators[idx]
            gen_idx = g.id
            scenario_arrays['generator_costs'][0, gen_idx] = g.cost.capex_p
            scenario_arrays['generator_costs'][2, gen_idx] = g.cost.fom
            scenario_arrays['generator_costs'][3, gen_idx] = g.cost.vom
            scenario_arrays['generator_costs'][4, gen_idx] = g.cost.lifetime
            scenario_arrays['generator_costs'][5, gen_idx] = g.cost.discount_rate
            scenario_arrays['generator_costs'][6, gen_idx] = g.cost.fuel_cost_mwh
            scenario_arrays['generator_costs'][7, gen_idx] = g.cost.fuel_cost_h
            scenario_arrays['generator_unit_size'][gen_idx] = g.unit_size

        for i, idx in enumerate(self.scenario.storages):
            s = self.scenario.storages[idx]
            storage_idx = s.id
            scenario_arrays['storage_costs'][0, storage_idx] = s.cost.capex_p
            scenario_arrays['storage_costs'][1, storage_idx] = s.cost.capex_e
            scenario_arrays['storage_costs'][2, storage_idx] = s.cost.fom
            scenario_arrays['storage_costs'][3, storage_idx] = s.cost.vom
            scenario_arrays['storage_costs'][4, storage_idx] = s.cost.lifetime
            scenario_arrays['storage_costs'][5, storage_idx] = s.cost.discount_rate
            
        for i, idx in enumerate(self.scenario.lines):
            l = self.scenario.lines[idx]
            line_idx = l.id
            scenario_arrays['line_costs'][0, line_idx] = l.cost.capex_p
            scenario_arrays['line_costs'][2, line_idx] = l.cost.fom
            scenario_arrays['line_costs'][3, line_idx] = l.cost.vom
            scenario_arrays['line_costs'][4, line_idx] = l.cost.lifetime
            scenario_arrays['line_costs'][5, line_idx] = l.cost.discount_rate
            scenario_arrays['line_costs'][6, line_idx] = l.cost.transformer_capex

        scenario_arrays['pv_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['solar'])]
        scenario_arrays['wind_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['wind'])]
        scenario_arrays['flexible_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['flexible'])]
        scenario_arrays['baseload_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['baseload'])]
        scenario_arrays['storage_cost_ids'] = scenario_arrays['storage_ids']
        scenario_arrays['line_cost_ids'] = scenario_arrays['line_ids'][~np.isin(scenario_arrays['line_ids'], minor_line_ids)]
        
        scenario_arrays['CBaseload'] = scenario_arrays['generator_capacities'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['baseload'])[0]]

        return scenario_arrays
    
    def _initialise_callback(self):
        temp_dir = os.path.join("results", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "callback.csv"), 'w', newline='') as csvfile:
            csv.writer(csvfile)

        with open(os.path.join(temp_dir, "population.csv"), 'w', newline='') as csvfile:
            csv.writer(csvfile)

        with open(os.path.join(temp_dir, "population_energies.csv"), 'w', newline='') as csvfile:
            csv.writer(csvfile)

    def _single_time(self):
        scenario_arrays = self._prepare_scenario_arrays()
        self._initialise_callback()

        self.result = differential_evolution(
            x0=self.decision_x0,
            func=parallel_wrapper, 
            bounds=list(zip(self.lower_bounds, self.upper_bounds)), 
            args=(scenario_arrays["MLoad"],
                    scenario_arrays["TSPV"],
                    scenario_arrays["TSWind"],
                    scenario_arrays["TSBaseload"],
                    scenario_arrays["network"],
                    scenario_arrays['transmission_mask'],
                    scenario_arrays["intervals"],
                    scenario_arrays["nodes"],
                    scenario_arrays["lines"],
                    scenario_arrays["years"],
                    scenario_arrays["resolution"],
                    scenario_arrays["allowance"],
                    scenario_arrays["generator_ids"],
                    scenario_arrays["generator_costs"],
                    scenario_arrays["storage_ids"],
                    scenario_arrays["storage_nodes"],
                    scenario_arrays["flexible_ids"],
                    scenario_arrays["storage_durations"],
                    scenario_arrays["storage_costs"],
                    scenario_arrays["line_ids"],
                    scenario_arrays["line_lengths"],
                    scenario_arrays["line_costs"],
                    scenario_arrays["TLoss"],
                    scenario_arrays["pv_idx"],
                    scenario_arrays["wind_idx"],
                    scenario_arrays["flexible_p_idx"],
                    scenario_arrays['storage_p_idx'],
                    scenario_arrays["storage_e_idx"],
                    scenario_arrays["lines_idx"],
                    scenario_arrays["solar_nodes"],
                    scenario_arrays["wind_nodes"],
                    scenario_arrays["flexible_nodes"],
                    scenario_arrays["baseload_nodes"],
                    scenario_arrays["CBaseload"],
                    scenario_arrays["pv_cost_ids"],
                    scenario_arrays["wind_cost_ids"],
                    scenario_arrays["flexible_cost_ids"],
                    scenario_arrays["baseload_cost_ids"],
                    scenario_arrays["storage_cost_ids"],
                    scenario_arrays["line_cost_ids"],
                    scenario_arrays["networksteps"],
                    scenario_arrays["storage_d_efficiencies"],
                    scenario_arrays["storage_c_efficiencies"],
                    scenario_arrays['Flexible_Limits_Annual'],
                    self.scenario.first_year,
                    scenario_arrays['generator_line_ids'],
                    scenario_arrays['storage_line_ids'],
                    scenario_arrays['generator_unit_size']
            ),
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
    
        scenario_arrays = self._prepare_scenario_arrays()
        args=(scenario_arrays["MLoad"],
                scenario_arrays["TSPV"],
                scenario_arrays["TSWind"],
                scenario_arrays["TSBaseload"],
                scenario_arrays["network"],
                scenario_arrays['transmission_mask'],
                scenario_arrays["intervals"],
                scenario_arrays["nodes"],
                scenario_arrays["lines"],
                scenario_arrays["years"],
                scenario_arrays["resolution"],
                scenario_arrays["allowance"],
                scenario_arrays["generator_ids"],
                scenario_arrays["generator_costs"],
                scenario_arrays["storage_ids"],
                scenario_arrays["storage_nodes"],
                scenario_arrays["flexible_ids"],
                scenario_arrays["storage_durations"],
                scenario_arrays["storage_costs"],
                scenario_arrays["line_ids"],
                scenario_arrays["line_lengths"],
                scenario_arrays["line_costs"],
                scenario_arrays["TLoss"],
                scenario_arrays["pv_idx"],
                scenario_arrays["wind_idx"],
                scenario_arrays["flexible_p_idx"],
                scenario_arrays['storage_p_idx'],
                scenario_arrays["storage_e_idx"],
                scenario_arrays["lines_idx"],
                scenario_arrays["solar_nodes"],
                scenario_arrays["wind_nodes"],
                scenario_arrays["flexible_nodes"],
                scenario_arrays["baseload_nodes"],
                scenario_arrays["CBaseload"],
                scenario_arrays["pv_cost_ids"],
                scenario_arrays["wind_cost_ids"],
                scenario_arrays["flexible_cost_ids"],
                scenario_arrays["baseload_cost_ids"],
                scenario_arrays["storage_cost_ids"],
                scenario_arrays["line_cost_ids"],
                scenario_arrays["networksteps"],
                scenario_arrays["storage_d_efficiencies"],
                scenario_arrays["storage_c_efficiencies"],
                scenario_arrays['Flexible_Limits_Annual'],
                self.scenario.first_year,
                scenario_arrays['generator_line_ids'],
                scenario_arrays['storage_line_ids'],
                scenario_arrays['generator_unit_size']
        )
    
        for group_key, idx_list in groups.items():
            
            self.scenario.logger.info(f"[near_optimum] exploring group '{group_key}'")
            
            def obj_batch_min(X): # 2-D array to allow vectorized DE
                batch = parallel_wrapper_lp(X, *args)
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
                batch = parallel_wrapper_lp(X, *args)
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
                
        scenario_arrays = self._prepare_scenario_arrays()
        args=(scenario_arrays["MLoad"],
                scenario_arrays["TSPV"],
                scenario_arrays["TSWind"],
                scenario_arrays["TSBaseload"],
                scenario_arrays["network"],
                scenario_arrays['transmission_mask'],
                scenario_arrays["intervals"],
                scenario_arrays["nodes"],
                scenario_arrays["lines"],
                scenario_arrays["years"],
                scenario_arrays["resolution"],
                scenario_arrays["allowance"],
                scenario_arrays["generator_ids"],
                scenario_arrays["generator_costs"],
                scenario_arrays["storage_ids"],
                scenario_arrays["storage_nodes"],
                scenario_arrays["flexible_ids"],
                scenario_arrays["storage_durations"],
                scenario_arrays["storage_costs"],
                scenario_arrays["line_ids"],
                scenario_arrays["line_lengths"],
                scenario_arrays["line_costs"],
                scenario_arrays["TLoss"],
                scenario_arrays["pv_idx"],
                scenario_arrays["wind_idx"],
                scenario_arrays["flexible_p_idx"],
                scenario_arrays['storage_p_idx'],
                scenario_arrays["storage_e_idx"],
                scenario_arrays["lines_idx"],
                scenario_arrays["solar_nodes"],
                scenario_arrays["wind_nodes"],
                scenario_arrays["flexible_nodes"],
                scenario_arrays["baseload_nodes"],
                scenario_arrays["CBaseload"],
                scenario_arrays["pv_cost_ids"],
                scenario_arrays["wind_cost_ids"],
                scenario_arrays["flexible_cost_ids"],
                scenario_arrays["baseload_cost_ids"],
                scenario_arrays["storage_cost_ids"],
                scenario_arrays["line_cost_ids"],
                scenario_arrays["networksteps"],
                scenario_arrays["storage_d_efficiencies"],
                scenario_arrays["storage_c_efficiencies"],
                scenario_arrays['Flexible_Limits_Annual'],
                self.scenario.first_year,
                scenario_arrays['generator_line_ids'],
                scenario_arrays['storage_line_ids'],
                scenario_arrays['generator_unit_size']
        )
        
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
                        batch = parallel_wrapper_lp(X, *args)
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
        scenario_arrays = self._prepare_scenario_arrays()
        solution = Solution_SingleTime(result_x,
                    scenario_arrays["MLoad"],
                    scenario_arrays["TSPV"],
                    scenario_arrays["TSWind"],
                    scenario_arrays["TSBaseload"],
                    scenario_arrays["network"],
                    scenario_arrays['transmission_mask'],
                    scenario_arrays["intervals"],
                    scenario_arrays["nodes"],
                    scenario_arrays["lines"],
                    scenario_arrays["years"],
                    scenario_arrays["resolution"],
                    scenario_arrays["allowance"],
                    scenario_arrays["generator_ids"],
                    scenario_arrays["generator_costs"],
                    scenario_arrays["storage_ids"],
                    scenario_arrays["storage_nodes"],
                    scenario_arrays["flexible_ids"],
                    scenario_arrays["storage_durations"],
                    scenario_arrays["storage_costs"],
                    scenario_arrays["line_ids"],
                    scenario_arrays["line_lengths"],
                    scenario_arrays["line_costs"],
                    scenario_arrays["TLoss"],
                    scenario_arrays["pv_idx"],
                    scenario_arrays["wind_idx"],
                    scenario_arrays["flexible_p_idx"],
                    scenario_arrays['storage_p_idx'],
                    scenario_arrays["storage_e_idx"],
                    scenario_arrays["lines_idx"],
                    scenario_arrays["solar_nodes"],
                    scenario_arrays["wind_nodes"],
                    scenario_arrays["flexible_nodes"],
                    scenario_arrays["baseload_nodes"],
                    scenario_arrays["CBaseload"],
                    scenario_arrays["pv_cost_ids"],
                    scenario_arrays["wind_cost_ids"],
                    scenario_arrays["flexible_cost_ids"],
                    scenario_arrays["baseload_cost_ids"],
                    scenario_arrays["storage_cost_ids"],
                    scenario_arrays["line_cost_ids"],
                    scenario_arrays["networksteps"],
                    scenario_arrays["storage_d_efficiencies"],
                    scenario_arrays["storage_c_efficiencies"],
                    scenario_arrays['Flexible_Limits_Annual'],
                    self.scenario.first_year,
                    scenario_arrays['generator_line_ids'],
                    scenario_arrays['storage_line_ids'],
                    scenario_arrays['generator_unit_size']
                    )
        solution.evaluate()

        return solution

    def evaluate(self):
        if self.config.type not in ['single_time','capacity_expansion','near_optimum','midpoint_explore']:
            raise Exception("Model type in config must be 'single_time' or 'capacity_expansion' or 'near_optimum' or 'midpoint_explore'")

        if self.config.type == 'single_time':
            self._single_time()
            
        elif self.config.type == 'near_optimum':
            self.optimal_lcoe = self.config.global_optimal_lcoe
            self.find_near_optimal_band()
            
        elif self.config.type == 'midpoint_explore':
            self.optimal_lcoe = self.config.global_optimal_lcoe
            self.explore_midpoints(self.config.midpoint_count)
            
        elif self.config.type == 'capacity_expansion':
            self._capacity_expansion()

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


