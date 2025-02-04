import numpy as np
from scipy.optimize import differential_evolution
from firm_ce.optimisation.single_time import parallel_wrapper
from firm_ce.file_manager import read_initial_guess
import csv

class Solver:
    def __init__(self, config, scenario) -> None:
        self.config = config
        self.scenario = scenario
        self.decision_x0 = read_initial_guess()
        print(self.decision_x0.shape)
        self.lower_bounds, self.upper_bounds = self._get_bounds()
        self.solution = None

    def _get_bounds(self):
        solar_generators = [self.scenario.generators[idx] for idx in self.scenario.generators if self.scenario.generators[idx].unit_type == 'solar']
        wind_generators = [self.scenario.generators[idx] for idx in self.scenario.generators if self.scenario.generators[idx].unit_type == 'wind']
        storages = [self.scenario.storages[idx] for idx in self.scenario.storages]
        lines = [self.scenario.lines[idx] for idx in self.scenario.lines]

        solar_lb = [generator.capacity + generator.min_build for generator in solar_generators]
        wind_lb = [generator.capacity + generator.min_build for generator in wind_generators]
        storage_p_lb = [0,0,0,0,0] ##### The storage.power_capacity needs to be fixed due to permutations
        storage_e_lb = [0,0,0,0,0]
        #storage_p_lb = [storage.power_capacity + storage.min_build_p for storage in storages]
        #storage_e_lb = [storage.energy_capacity + storage.min_build_e for storage in storages] # For nodes
        line_lb = [line.capacity + line.min_build for line in lines]
        storage_e_W_cutoffs_lb = (len(self.scenario.storages)-len(self.scenario.nodes_with_storage))*[0.0]
        lower_bounds = np.array(solar_lb + wind_lb + storage_p_lb + storage_e_lb + line_lb + storage_e_W_cutoffs_lb)

        solar_ub = [generator.capacity + generator.max_build for generator in solar_generators]
        wind_ub = [generator.capacity + generator.max_build for generator in wind_generators]
        storage_p_ub = [1000,1000,1000,1000,1000] ##### The storage.power_capacity needs to be fixed due to permutations
        storage_e_ub = [10000,10000,10000,10000,10000]
        #storage_p_ub = [storage.power_capacity + storage.max_build_p for storage in storages] 
        #storage_e_ub = [storage.energy_capacity + storage.max_build_e if storage.duration > 0 else 0.0 for storage in storages] # For nodes with storage else 0
        line_ub = [line.capacity + line.max_build for line in lines]
        storage_e_W_cutoffs_ub = (len(self.scenario.storages)-len(self.scenario.nodes_with_storage))*[self.scenario.max_frequency]
        upper_bounds = np.array(solar_ub + wind_ub + storage_p_ub + storage_e_ub + line_ub + storage_e_W_cutoffs_ub)

        return lower_bounds, upper_bounds

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
            (8, max(self.scenario.generators)+1), dtype=np.float64
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

        scenario_arrays['storage_power_capacities'] = np.array(
            [self.scenario.storages[idx].power_capacity
            for idx in self.scenario.storages],
            dtype=np.float64
        )

        scenario_arrays['storage_energy_capacities'] = np.array(
            [self.scenario.storages[idx].energy_capacity
            for idx in self.scenario.storages],
            dtype=np.float64
        )

        scenario_arrays['storage_unit_types'] = np.array(
            [scenario_arrays['storage_unit_types_setting'][self.scenario.storages[idx].unit_type]
            for idx in self.scenario.storages],
            dtype=np.int64
        )

        scenario_arrays['max_frequency'] = self.scenario.max_frequency
        scenario_arrays["nodes_with_storage"] = np.array(list(set(scenario_arrays['storage_nodes'])), dtype=np.int64)

        scenario_arrays['storage_durations'] = np.array(
            [self.scenario.storages[idx].duration
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
        scenario_arrays['storage_p_idx'] = scenario_arrays['wind_idx'] + len(scenario_arrays['nodes_with_storage'])
        scenario_arrays['storage_e_idx'] = scenario_arrays['storage_p_idx'] + len(scenario_arrays['nodes_with_storage'])
        scenario_arrays['lines_idx'] = scenario_arrays['storage_e_idx'] + len(self.scenario.lines)
        scenario_arrays['storage_e_W_idx'] = scenario_arrays['lines_idx'] + (len(self.scenario.storages)-len(scenario_arrays['nodes_with_storage']))

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
            scenario_arrays['generator_costs'][0, i] = g.cost.capex_p
            scenario_arrays['generator_costs'][2, i] = g.cost.fom
            scenario_arrays['generator_costs'][3, i] = g.cost.vom
            scenario_arrays['generator_costs'][4, i] = g.cost.lifetime
            scenario_arrays['generator_costs'][5, i] = g.cost.discount_rate
            scenario_arrays['generator_costs'][7, i] = g.cost.lcoe

        for i, idx in enumerate(self.scenario.storages):
            s = self.scenario.storages[idx]
            scenario_arrays['storage_costs'][0, i] = s.cost.capex_p
            scenario_arrays['storage_costs'][1, i] = s.cost.capex_e
            scenario_arrays['storage_costs'][2, i] = s.cost.fom
            scenario_arrays['storage_costs'][3, i] = s.cost.vom
            scenario_arrays['storage_costs'][4, i] = s.cost.lifetime
            scenario_arrays['storage_costs'][5, i] = s.cost.discount_rate
            
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

        scenario_arrays['solar_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['solar'])[0]] 
        scenario_arrays['wind_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['wind'])[0]] 
        scenario_arrays['flexible_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['flexible'])[0]] 
        scenario_arrays['baseload_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == scenario_arrays['generator_unit_types_setting']['baseload'])[0]] 
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
                    scenario_arrays["generator_nodes"],
                    scenario_arrays["generator_capacities"],
                    scenario_arrays["generator_costs"],
                    scenario_arrays["generator_unit_types"],
                    scenario_arrays["storage_ids"],
                    scenario_arrays["storage_nodes"],
                    scenario_arrays["storage_power_capacities"],
                    scenario_arrays["storage_energy_capacities"],
                    scenario_arrays["storage_unit_types"],
                    scenario_arrays["nodes_with_storage"],
                    scenario_arrays['max_frequency'],
                    scenario_arrays["storage_durations"],
                    scenario_arrays["storage_costs"],
                    scenario_arrays["line_ids"],
                    scenario_arrays["line_lengths"],
                    scenario_arrays["line_costs"],
                    scenario_arrays["TLoss"],
                    scenario_arrays["pv_idx"],
                    scenario_arrays["wind_idx"],
                    scenario_arrays["storage_p_idx"],
                    scenario_arrays["storage_e_idx"],
                    scenario_arrays["lines_idx"],
                    scenario_arrays['storage_e_W_idx'],
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


