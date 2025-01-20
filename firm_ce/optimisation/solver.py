from typing import Dict, List, Tuple
import numpy as np
from numba import njit, prange
from scipy.optimize import differential_evolution

from firm_ce.network import get_transmission_flows_t
from firm_ce.constants import TRIANGULAR
from firm_ce.components import calculate_costs

class Solution_SingleTime:
    def __init__(self, x, scenario_arrays) -> None:
        self.x = x  
        self.evaluated=False   
        self.lcoe = 0.0
        self.penalties = 0.0

        self.MLoad = scenario_arrays['MLoad']
        self.intervals = scenario_arrays['intervals']
        self.nodes = scenario_arrays['nodes']
        self.lines = scenario_arrays['lines']
        self.efficiency = scenario_arrays['efficiency'] ########## BASE THIS ON STORAGE-LEVEL EFFICIENCIES?
        self.resolution = scenario_arrays['resolution']
        self.years = scenario_arrays['years']
        self.energy = self.MLoad.sum() * self.resolution / self.years
        self.allowance = scenario_arrays['allowance']

        self.unit_types = scenario_arrays['unit_types']

        # Generators
        self.generator_ids = scenario_arrays['generator_ids'] 
        self.generator_nodes = scenario_arrays['generator_nodes'] 
        self.generator_capacities = scenario_arrays['generator_capacities'] 
        self.generator_annual_generation = scenario_arrays['generator_annual_generation'] 
        self.generator_unit_types = scenario_arrays['generator_unit_types'] 
        self.generator_costs = scenario_arrays['generator_costs']

        self.TSPV = scenario_arrays['TSPV'] 
        self.TSWind = scenario_arrays['TSWind'] 
        self.CPeak = self.generator_capacities[np.where(self.generator_unit_types == self.unit_types['flexible'])[0]]
        self.CBaseload = self.generator_capacities[np.where(self.generator_unit_types == self.unit_types['baseload'])[0]]
        self.GBaseload = self.CBaseload * np.ones((self.intervals,len(self.CBaseload)), dtype=np.float64)
        self.GFlexible = np.zeros((self.intervals,len(self.CPeak)), dtype=np.float64)

        # Storages
        self.storage_ids = scenario_arrays['storage_ids'] 
        self.storage_nodes = scenario_arrays['storage_nodes'] 
        self.storage_power_capacities = scenario_arrays['storage_power_capacities'] 
        self.storage_energy_capacities = scenario_arrays['storage_energy_capacities'] 
        self.storage_annual_discharge = scenario_arrays['storage_annual_discharge'] 
        self.storage_costs = scenario_arrays['storage_costs']

        self.Discharge = scenario_arrays['storage_discharges']
        self.Charge = scenario_arrays['storage_charges'] 

        # Lines
        self.line_ids = scenario_arrays['line_ids'] 
        self.line_lengths = scenario_arrays['line_lengths']
        self.line_annual_TFlowsAbs = scenario_arrays['line_annual_TFlowsAbs'] 
        self.line_costs = scenario_arrays['line_costs']
        
        self.network = scenario_arrays['network'] 
        self.TLoss = scenario_arrays['TLoss'] 
        self.TFlows = scenario_arrays['TFlows'] 
        self.TFlowsAbs = scenario_arrays['TFlowsAbs'] 

        # Decision Variables
        self.CPV = x[: scenario_arrays['pv_idx']]
        self.CWind = x[scenario_arrays['pv_idx'] : scenario_arrays['wind_idx']]
        self.CPHP = x[scenario_arrays['wind_idx'] : scenario_arrays['storage_p_idx']]
        self.CPHS = x[scenario_arrays['storage_p_idx'] : scenario_arrays['storage_e_idx']]
        self.CTrans = x[scenario_arrays['storage_e_idx'] :] ###### THIS NEEDS TO BE TREATED AS A BOUND

        # Nodal Values
        self.solar_nodes = self.generator_nodes[np.where(self.generator_unit_types == self.unit_types['solar'])[0]] 
        self.wind_nodes = self.generator_nodes[np.where(self.generator_unit_types == self.unit_types['wind'])[0]] 
        self.flexible_nodes = self.generator_nodes[np.where(self.generator_unit_types == self.unit_types['flexible'])[0]] 
        self.baseload_nodes = self.generator_nodes[np.where(self.generator_unit_types == self.unit_types['baseload'])[0]] 
        self.storage_nodes = self.storage_nodes

        self.CPHP_nodal = np.zeros(self.nodes, dtype=np.float64)
        self.CPHS_nodal = np.zeros(self.nodes, dtype=np.float64)
        for idx in self.storage_nodes:
            mask = (self.storage_nodes == idx)
            self.CPHP_nodal[idx] = self.CPHP[mask].sum() 
            self.CPHS_nodal[idx] = self.CPHS[mask].sum()            

        self.GPV = self.CPV[np.newaxis, :] * scenario_arrays['TSPV'] * 1000
        self.GPV_nodal = np.zeros((self.intervals, self.nodes), dtype=np.float64) 
        for idx in self.solar_nodes:
            mask = (self.solar_nodes == idx)
            self.GPV_nodal[:,idx] = self.GPV[:,mask].sum(axis=1)

        self.GWind = self.CWind[np.newaxis, :] * scenario_arrays['TSWind'] * 1000 
        self.GWind_nodal = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        for idx in self.wind_nodes:
            mask = (self.wind_nodes == idx)
            self.GWind_nodal[:,idx] = self.GWind[:,mask].sum(axis=1)

        self.CPeak_nodal = np.zeros(self.nodes, dtype=np.float64)
        for idx in self.flexible_nodes:
            mask = (self.flexible_nodes == idx)
            self.CPeak_nodal[idx] = self.CPeak[mask].sum()   
        
        self.CBaseload_nodal = np.zeros(self.nodes, dtype=np.float64)
        self.GBaseload_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64) 
        for idx in self.baseload_nodes:
            mask = (self.baseload_nodes == idx)
            self.CBaseload_nodal[idx] = self.CBaseload[mask].sum()  
            self.GBaseload_nodal[:,idx] = self.GBaseload[:,mask].sum(axis=1)

        self.GFlexible_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Spillage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Charge_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Discharge_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Storage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Deficit_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Import_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64) 
        self.Export_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64)   

    def _reliability(self, flexible, start=None, end=None):
        network = self.network
        networksteps = np.where(TRIANGULAR == network.shape[2])[0][0]
        
        Netload = (self.MLoad - self.GPV - self.GWind - self.GBaseload_nodal)[start:end]
        Netload -= flexible
        shape2d = intervals, nodes = len(Netload), self.nodes

        Pcapacity = self.CPHP_nodal * 1000
        Scapacity = self.CPHS_nodal * 1000

        Hcapacity = self.CTrans * 1000 # GW to MW
        ntrans = len(self.CTrans)
        efficiency, resolution = self.efficiency, self.resolution 

        Discharge = np.zeros(shape2d, dtype=np.float64)
        Charge = np.zeros(shape2d, dtype=np.float64)
        Storage = np.zeros(shape2d, dtype=np.float64)
        Deficit = np.zeros(shape2d, dtype=np.float64)
        Transmission = np.zeros((intervals, ntrans, nodes), dtype = np.float64)
        Storaget_1 = 0.5*Scapacity

        for t in range(intervals):
            Netloadt = Netload[t]

            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
            Deficitt = np.maximum(Netloadt - Discharget ,0)

            Transmissiont=np.zeros((ntrans, nodes), dtype=np.float64)
        
            if Deficitt.sum() > 1e-6:
                # raise KeyboardInterrupt
                # Fill deficits with transmission allowing drawing down from neighbours battery reserves
                Surplust = -1 * np.minimum(0, Netloadt) + (np.minimum(Pcapacity, Storaget_1 / resolution) - Discharget)
                Transmissiont = get_transmission_flows_t(Deficitt, Surplust, Hcapacity, network, networksteps, 
                                    np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))
                
                Netloadt = Netload[t] - Transmissiont.sum(axis=0)
                Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
            
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Surplust = -1 * np.minimum(0, Netloadt + Charget)# charge itself first, then distribute
            if Surplust.sum() > 1e-6:
                # raise KeyboardInterrupt
                # Distribute surplus energy with transmission to areas with spare charging capacity
                Fillt = (Discharget # load not met by gen and transmission
                        + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #full charging capacity
                        - Charget) #charge capacity already in use
                Transmissiont = get_transmission_flows_t(Fillt, Surplust, Hcapacity, network, networksteps,
                                    np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))
                #print(Netload[t])
                Netloadt = Netload[t] - Transmissiont.sum(axis=0)
                Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
                Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

            Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
            Storaget_1 = Storaget.copy()
            
            Discharge[t] = Discharget
            Charge[t] = Charget
            Storage[t] = Storaget
            Transmission[t] = Transmissiont
        
        ImpExp = Transmission.sum(axis=1)
        
        Deficit = np.maximum(0, Netload - ImpExp - Discharge)
        Spillage = -1 * np.minimum(0, Netload - ImpExp + Charge)

        self.Spillage_nodal = Spillage
        self.Charge_nodal = Charge
        self.Discharge_nodal = Discharge
        self.Storage_nodal = Storage
        self.Deficit_nodal = Deficit
        self.Import_nodal = np.maximum(0, ImpExp)
        self.Export_nodal = -1 * np.minimum(0, ImpExp)

        self.TFlows = (Transmission).sum(axis=2)
        
        return Deficit

    def _calculate_costs(self):
        solution_cost = calculate_costs(self)
        return solution_cost
    
    def _apportion_nodal_generation(self):
        for idx in range(0,len(self.flexible_nodes)):
            mask = (self.flexible_nodes == self.flexible_nodes[idx])
            self.GFlexible[:,idx] = self.CPeak[idx] / self.CPeak[mask].sum() * self.GFlexible_nodal[:,self.flexible_nodes[idx]] if self.CPeak[mask].sum() > 0 else self.GFlexible_nodal[:,self.flexible_nodes[idx]]
        
        for idx in range(0,len(self.baseload_nodes)):
            mask = (self.baseload_nodes == self.baseload_nodes[idx])
            self.GBaseload[:,idx] = self.CBaseload[idx] / self.CBaseload[mask].sum() * self.GBaseload_nodal[:,self.baseload_nodes[idx]] if self.CBaseload[mask].sum() > 0 else self.GBaseload_nodal[:,self.baseload_nodes[idx]]

        for idx in range(0,len(self.storage_nodes)):
            mask = (self.storage_nodes == self.storage_nodes[idx])
            self.Discharge[:,idx] = self.CPHP[idx] / self.CPHP[mask].sum() * self.Discharge_nodal[:,self.storage_nodes[idx]] if self.CPHP[mask].sum() > 0 else self.Discharge_nodal[:,self.storage_nodes[idx]]

        return None

    def _calculate_annual_generation(self):
        self.GPV_annual = self.GPV.sum(axis=0) * self.resolution / self.years
        self.GWind_annual = self.GWind.sum(axis=0) * self.resolution / self.years
        self.GFlexible_annual = self.GFlexible.sum(axis=0) * self.resolution / self.efficiency / self.years
        self.GBaseload_annual = self.GBaseload.sum(axis=0) * self.resolution / self.years
        self.GDischarge_annual = self.Discharge.sum(axis=0) * self.resolution / self.years
        self.TFlowsAbs_annual = self.TFlowsAbs.sum(axis=0) * self.resolution / self.years

        return None

    def _objective(self) -> List[float]:
        deficit = self._reliability(flexible=np.zeros((self.intervals, self.nodes), dtype=np.float64))      
        self.GFlexible_nodal = deficit * self.resolution / self.years / (0.5 * (1 + self.efficiency))

        deficit = self._reliability(flexible=self.GFlexible_nodal)
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution - self.allowance) * 1000000 # MWh
        self.TFlowsAbs = np.abs(self.TFlows)

        self._apportion_nodal_generation()
        self._calculate_annual_generation()
        cost = self._calculate_costs()

        loss = self.TFlowsAbs.sum(axis=0) * self.TLoss
        loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - loss)

        print(lcoe, cost, self.energy, loss)

        return lcoe, pen_deficit

    def evaluate(self):
        self.lcoe, self.penalties = self._objective()
        self.evaluated=True 

class Solver:
    def __init__(self, config, scenario) -> None:
        self.config = config
        self.scenario = scenario
        self.decision_x0 = None
        self.lower_bounds, self.upper_bounds = self._get_bounds()
        self.result = None

    def _get_bounds(self):
        solar_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)+1) if self.scenario.generators[idx].unit_type == 'solar']
        wind_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)+1) if self.scenario.generators[idx].unit_type == 'wind']
        storages = [self.scenario.storages[idx] for idx in range(0,max(self.scenario.storages)+1)]
        lines = [self.scenario.lines[idx] for idx in range(0,max(self.scenario.lines)+1)]

        solar_lb = [generator.capacity + generator.min_build for generator in solar_generators]
        wind_lb = [generator.capacity + generator.min_build for generator in wind_generators]
        storage_p_lb = [storage.power_capacity + storage.min_build_p for storage in storages]
        storage_e_lb = [storage.energy_capacity + storage.min_build_e for storage in storages]
        line_lb = [line.capacity + line.min_build for line in lines]
        lower_bounds = np.array(solar_lb + wind_lb + storage_p_lb + storage_e_lb + line_lb)

        solar_ub = [generator.capacity + generator.max_build for generator in solar_generators]
        wind_ub = [generator.capacity + generator.max_build for generator in wind_generators]
        storage_p_ub = [storage.power_capacity + storage.max_build_p for storage in storages]
        storage_e_ub = [storage.energy_capacity + storage.max_build_e for storage in storages]
        line_ub = [line.capacity + line.max_build for line in lines]
        upper_bounds = np.array(solar_ub + wind_ub + storage_p_ub + storage_e_ub + line_ub)

        return lower_bounds, upper_bounds

    def _prepare_scenario_arrays(self):
        
        scenario_arrays = {}
        node_names = {self.scenario.nodes[idx].name : self.scenario.nodes[idx].id for idx in self.scenario.nodes}

        # Static parameters
        scenario_arrays['MLoad'] = np.array([self.scenario.nodes[idx].demand_data for idx in range(0,max(self.scenario.nodes)+1)], dtype=np.float64).T
        scenario_arrays['intervals'], scenario_arrays['nodes'] = scenario_arrays['MLoad'].shape
        scenario_arrays['lines'] = len(self.scenario.lines)
        scenario_arrays['years'] = self.scenario.final_year - self.scenario.first_year   
        scenario_arrays['efficiency'] = 0.8
        scenario_arrays['resolution'] = self.scenario.resolution
        scenario_arrays['allowance'] = self.scenario.allowance

        # Unit types
        scenario_arrays['unit_types'] = {'solar': 0,
                                        'wind': 1,
                                        'flexible': 2,
                                        'baseload': 3
                                        } ############ GENERATE THIS IN SCENARIOS
        
        # Generators
        scenario_arrays['generator_ids'] = np.array([self.scenario.generators[idx].id for idx in self.scenario.generators], dtype=np.int16)
        scenario_arrays['generator_nodes'] = np.array([node_names[self.scenario.generators[idx].node] for idx in self.scenario.generators], dtype=np.int16)
        scenario_arrays['generator_capacities'] = np.array([self.scenario.generators[idx].capacity for idx in self.scenario.generators], dtype=np.float64)
        scenario_arrays['generator_annual_generation'] = np.zeros(len(self.scenario.generators), dtype=np.float64)
        scenario_arrays['generator_unit_types'] = np.array([scenario_arrays['unit_types'][self.scenario.generators[idx].unit_type] for idx in self.scenario.generators], dtype=np.int16)
        scenario_arrays['generator_costs'] = np.zeros((7,len(self.scenario.generators)), dtype=np.float64)

        scenario_arrays['TSPV'] = np.array([self.scenario.generators[idx].data for idx in self.scenario.generators if self.scenario.generators[idx].unit_type == 'solar']).T 
        scenario_arrays['TSWind'] = np.array([self.scenario.generators[idx].data for idx in self.scenario.generators if self.scenario.generators[idx].unit_type == 'wind']).T
        
        # Storages
        scenario_arrays['storage_ids'] = np.array([self.scenario.storages[idx].id for idx in self.scenario.storages], dtype=np.int16)
        scenario_arrays['storage_nodes'] = np.array([node_names[self.scenario.storages[idx].node] for idx in self.scenario.storages], dtype=np.int16)
        scenario_arrays['storage_power_capacities'] = np.array([self.scenario.storages[idx].power_capacity for idx in self.scenario.storages], dtype=np.float64)
        scenario_arrays['storage_energy_capacities'] = np.array([self.scenario.storages[idx].energy_capacity for idx in self.scenario.storages], dtype=np.float64)
        scenario_arrays['storage_annual_discharge'] = np.zeros(len(self.scenario.storages), dtype=np.float64)
        scenario_arrays['storage_costs'] = np.zeros((7,len(self.scenario.storages)), dtype=np.float64)

        scenario_arrays['storage_discharges'] = np.zeros((scenario_arrays['intervals'],len(self.scenario.storages)), dtype=np.float64)
        scenario_arrays['storage_charges'] = np.zeros((scenario_arrays['intervals'],len(self.scenario.storages)), dtype=np.float64)

        # Lines
        scenario_arrays['line_ids'] = np.array([self.scenario.lines[idx].id for idx in self.scenario.lines], dtype=np.int16)
        scenario_arrays['line_capacities'] = np.array([self.scenario.lines[idx].capacity for idx in self.scenario.lines], dtype=np.float64)
        scenario_arrays['line_lengths'] = np.array([self.scenario.lines[idx].length for idx in self.scenario.lines], dtype=np.int32)
        scenario_arrays['line_annual_TFlowsAbs'] = np.array([0.0 for idx in self.scenario.lines], dtype=np.float64)
        scenario_arrays['line_costs'] =  np.zeros((7,len(self.scenario.lines)), dtype=np.float64)
        
        scenario_arrays['network'] = self.scenario.network.network  
        scenario_arrays['TLoss'] = np.array([self.scenario.lines[idx].loss_factor for idx in self.scenario.lines], dtype=np.float64)
        scenario_arrays['TFlows'] = np.zeros((scenario_arrays['intervals'],len(self.scenario.lines)), dtype=np.float64)
        scenario_arrays['TFlowsAbs'] = np.zeros((scenario_arrays['intervals'],len(self.scenario.lines)), dtype=np.float64) 

        # Decision variable indices
        scenario_arrays['pv_idx'] = scenario_arrays['TSPV'].shape[1]
        scenario_arrays['wind_idx'] = scenario_arrays['pv_idx'] + scenario_arrays['TSWind'].shape[1]
        scenario_arrays['storage_p_idx'] = scenario_arrays['wind_idx'] + len(self.scenario.storages)
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
            )
        '''       
        
        for idx in range(0,len(self.scenario.generators)):
            scenario_arrays['generator_costs'][0,idx] = self.scenario.generators[idx].cost.capex_p
            scenario_arrays['generator_costs'][2,idx] = self.scenario.generators[idx].cost.fom
            scenario_arrays['generator_costs'][3,idx] = self.scenario.generators[idx].cost.vom
            scenario_arrays['generator_costs'][4,idx] = self.scenario.generators[idx].cost.lifetime
            scenario_arrays['generator_costs'][5,idx] = self.scenario.generators[idx].cost.discount_rate

        for idx in range(0,len(self.scenario.storages)):
            scenario_arrays['storage_costs'][0,idx] = self.scenario.storages[idx].cost.capex_p
            scenario_arrays['storage_costs'][1,idx] = self.scenario.storages[idx].cost.capex_e
            scenario_arrays['storage_costs'][2,idx] = self.scenario.storages[idx].cost.fom
            scenario_arrays['storage_costs'][3,idx] = self.scenario.storages[idx].cost.vom
            scenario_arrays['storage_costs'][4,idx] = self.scenario.storages[idx].cost.lifetime
            scenario_arrays['storage_costs'][5,idx] = self.scenario.storages[idx].cost.discount_rate
            
        for idx in range(0,len(self.scenario.lines)):
            scenario_arrays['line_costs'][0,idx] = self.scenario.lines[idx].cost.capex_p
            scenario_arrays['line_costs'][2,idx] = self.scenario.lines[idx].cost.fom
            scenario_arrays['line_costs'][3,idx] = self.scenario.lines[idx].cost.vom
            scenario_arrays['line_costs'][4,idx] = self.scenario.lines[idx].cost.lifetime
            scenario_arrays['line_costs'][5,idx] = self.scenario.lines[idx].cost.discount_rate
            scenario_arrays['line_costs'][6,idx] = self.scenario.lines[idx].cost.transformer_capex
        
        return scenario_arrays

    def _single_time(self):
        scenario_arrays = self._prepare_scenario_arrays()
        wrapped_objective = parallel_wrapper_st_factory(scenario_arrays)

        self.result = differential_evolution(
            x0=self.decision_x0,
            func=wrapped_objective, 
            bounds=list(zip(self.lower_bounds, self.upper_bounds)), 
            tol=0,
            maxiter=self.config.iterations, 
            popsize=self.config.population, 
            mutation=(0.2,self.config.mutation), 
            recombination=self.config.recombination,
            disp=True, 
            polish=False, 
            updating='deferred',
            callback=None, 
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

def parallel_wrapper_st_factory(scenario_arrays):
    #@njit(parallel=True)
    def parallel_wrapper_st(xs):
        result = np.empty(xs.shape[1], dtype=np.float64)
        for i in prange(xs.shape[1]):
            result[i] = objective_st(xs[:,i], scenario_arrays)
        return result
    return parallel_wrapper_st

def objective_st(x, scenario_arrays):
    solution = Solution_SingleTime(x, scenario_arrays)
    solution.evaluate()
    return solution.lcoe + solution.penalties