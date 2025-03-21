import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from firm_ce.optimisation.single_time import parallel_wrapper
from firm_ce.file_manager import read_initial_guess
import csv

class Solver:
    def __init__(self, config, scenario) -> None:
        self.config = config
        self.scenario = scenario
        self.decision_x0 = read_initial_guess()
        self.lower_bounds, self.upper_bounds, self.x_W_offset, self.cutoff_W_per_node = self._get_bounds()
        self.solution = None

    def _get_cutoff_W_per_node(self, flexible_generators):
        cutoff_W_per_node = []
        for node in sorted(self.scenario.nodes_with_balancing):
            num_storages = len([self.scenario.storages[idx] for idx in self.scenario.storages if self.scenario.node_names[self.scenario.storages[idx].node] == node])
            num_flexible = len([generator for generator in flexible_generators if self.scenario.node_names[generator.node] == node])
            cutoff_W_per_node.append(num_storages + num_flexible - 1)
        return cutoff_W_per_node

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
        line_lb = [line.capacity + line.min_build for line in lines]
        lower_bounds = np.array(solar_lb + wind_lb + flexible_p_lb + storage_p_lb + storage_e_lb + line_lb)

        solar_ub = [generator.capacity + generator.max_build for generator in solar_generators]
        wind_ub = [generator.capacity + generator.max_build for generator in wind_generators]
        flexible_p_ub = [generator.capacity + generator.max_build for generator in flexible_generators] 
        storage_p_ub = [storage.power_capacity + storage.max_build_p for storage in storages] 
        storage_e_ub = [storage.energy_capacity + storage.max_build_e if storage.duration == 0 else 0.0 for storage in storages]
        line_ub = [line.capacity + line.max_build for line in lines]
        upper_bounds = np.array(solar_ub + wind_ub + flexible_p_ub + storage_p_ub + storage_e_ub + line_ub)

        x_W_offset = len(solar_lb) + len(wind_lb) + len(flexible_p_lb) + len(storage_p_lb) + len(storage_e_lb) + len(line_lb)

        cutoff_W_per_node = self._get_cutoff_W_per_node(flexible_generators)

        return lower_bounds, upper_bounds, x_W_offset, cutoff_W_per_node

    def _prepare_scenario_arrays(self):
        
        scenario_arrays = {}
        node_names = {self.scenario.nodes[idx].name : self.scenario.nodes[idx].id for idx in self.scenario.nodes}

        # Static parameters
        scenario_arrays['MLoad'] = np.array(
            [self.scenario.nodes[idx].demand_data
            for idx in range(0, max(self.scenario.nodes)+1)],
            dtype=np.float64
        ).T

        scenario_arrays['intervals'], scenario_arrays['nodes'] = scenario_arrays['MLoad'].shape
        scenario_arrays['lines'] = len(self.scenario.lines)
        scenario_arrays['years'] = self.scenario.final_year - self.scenario.first_year
        scenario_arrays['efficiency'] = 0.8
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

        scenario_arrays['generator_costs'] = np.zeros(
            (7, max(self.scenario.generators)+1), dtype=np.float64
        )

        scenario_arrays['TSPV'] = np.array([
            self.scenario.generators[idx].data
            for idx in self.scenario.generators
            if self.scenario.generators[idx].unit_type == 'solar'
        ], dtype=np.float64).T

        scenario_arrays['TSWind'] = np.array([
            self.scenario.generators[idx].data
            for idx in self.scenario.generators
            if self.scenario.generators[idx].unit_type == 'wind'
        ], dtype=np.float64).T

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
        scenario_arrays["nodes_with_balancing"] = np.array(list(set(np.concatenate([scenario_arrays['storage_nodes'], scenario_arrays['flexible_nodes']]))), dtype=np.int64)

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

        scenario_arrays['line_lengths'] = np.array(
            [self.scenario.lines[idx].length for idx in self.scenario.lines],
            dtype=np.float64
        )

        scenario_arrays['line_costs'] = np.zeros(
            (7, max(self.scenario.lines)+1), dtype=np.float64
        )

        scenario_arrays['network'] = self.scenario.network.network
        scenario_arrays['networksteps'] = self.scenario.network.networksteps

        scenario_arrays['TLoss'] = np.array(
            [self.scenario.lines[idx].loss_factor for idx in self.scenario.lines],
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
        scenario_arrays['balancing_W_idx'] = scenario_arrays['lines_idx'] + (len(self.scenario.storages)-len(scenario_arrays['nodes_with_balancing']))

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
            scenario_arrays['line_costs'][0, i] = l.cost.capex_p
            scenario_arrays['line_costs'][2, i] = l.cost.fom
            scenario_arrays['line_costs'][3, i] = l.cost.vom
            scenario_arrays['line_costs'][4, i] = l.cost.lifetime
            scenario_arrays['line_costs'][5, i] = l.cost.discount_rate
            scenario_arrays['line_costs'][6, i] = l.cost.transformer_capex

        scenario_arrays['pv_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['solar'])]
        scenario_arrays['wind_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['wind'])]
        scenario_arrays['flexible_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['flexible'])]
        scenario_arrays['baseload_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['baseload'])]
        scenario_arrays['storage_cost_ids'] = scenario_arrays['storage_ids']
        scenario_arrays['line_cost_ids'] = scenario_arrays['line_ids']

        scenario_arrays['CPeak'] = scenario_arrays['generator_capacities'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['flexible'])[0]]
        scenario_arrays['CBaseload'] = scenario_arrays['generator_capacities'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['baseload'])[0]]
        
        return scenario_arrays
    
    def _initialise_callback(self):
        with open('results/callback.csv', 'w', newline='') as csvfile:
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
                    scenario_arrays["network"],
                    scenario_arrays["intervals"],
                    scenario_arrays["nodes"],
                    scenario_arrays["lines"],
                    scenario_arrays["years"],
                    scenario_arrays["efficiency"],
                    scenario_arrays["resolution"],
                    scenario_arrays["allowance"],
                    scenario_arrays["generator_ids"],
                    scenario_arrays["generator_costs"],
                    scenario_arrays["storage_ids"],
                    scenario_arrays["storage_nodes"],
                    scenario_arrays["flexible_ids"],
                    scenario_arrays["nodes_with_balancing"],
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
                    scenario_arrays['balancing_W_idx'],
                    scenario_arrays["solar_nodes"],
                    scenario_arrays["wind_nodes"],
                    scenario_arrays["flexible_nodes"],
                    scenario_arrays["baseload_nodes"],
                    scenario_arrays["CPeak"],
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
        
    def _capacity_expansion(self):
        pass

    def evaluate(self):
        if self.config.type not in ['single_time','capacity_expansion']:
            raise Exception("Model type in config must be 'single_time' or 'capacity_expansion")

        if self.config.type == 'single_time':
            self.solution = self._single_time()
        elif self.config.type == 'capacity_expansion':
            self.solution = self._capacity_expansion()

def callback(xk, convergence=None):
    with open('results/callback.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(xk))


