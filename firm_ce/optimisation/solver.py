from typing import Dict, List, Tuple
import numpy as np
from numba import njit, prange
from scipy.optimize import differential_evolution

from firm_ce.components import SolutionCost
from firm_ce.network import get_transmission_flows_t
from firm_ce.constants import TRIANGULAR

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
        self.efficiency = scenario_arrays['efficiency']
        self.resolution = scenario_arrays['resolution']
        self.years = scenario_arrays['years']
        self.energy = self.MLoad.sum() * self.resolution / self.years
        #self.allowance = scenario_arrays['allowance']
        
        self.TLoss = scenario_arrays['TLoss']
        self.TFlows = scenario_arrays['TFlows']
        self.TFlowsAbs = scenario_arrays['TFlowsAbs']
        self.Discharge = scenario_arrays['Discharge']
        self.Charge = scenario_arrays['Charge']
        self.CPeak = scenario_arrays['CPeak']
        self.GBaseload = scenario_arrays['CBaseload'][np.newaxis, :] * np.ones((self.intervals,self.nodes), dtype=np.int64)
        self.network = scenario_arrays['network']
        
        self.CPV = x[: scenario_arrays['pv_idx']]
        self.CWind = x[scenario_arrays['pv_idx'] : scenario_arrays['wind_idx']]
        CPHP = x[scenario_arrays['wind_idx'] : scenario_arrays['storage_p_idx']]
        CPHS = x[scenario_arrays['storage_p_idx'] : scenario_arrays['storage_e_idx']]
        self.CTrans = x[scenario_arrays['storage_e_idx'] :]

        GPV = self.CPV[np.newaxis, :] * scenario_arrays['TSPV'] * 1000
        GWind = self.CWind[np.newaxis, :] * scenario_arrays['TSWind'] * 1000   


        self.CPHP = np.zeros(self.nodes, dtype=np.float64)
        for val, idx in zip(CPHP, scenario_arrays['storage_nodes']):
            self.CPHP[idx] += val

        self.CPHS = np.zeros(self.nodes, dtype=np.float64)
        for val, idx in zip(CPHS, scenario_arrays['storage_nodes']):
            self.CPHS[idx] += val

        self.GPV = np.zeros((self.intervals, self.nodes), dtype=np.float64) 
        for idx in scenario_arrays['pv_nodes']:
            mask = (scenario_arrays['pv_nodes'] == idx)
            self.GPV[:,idx] = GPV[:,mask].sum(axis=1)
        
        self.GWind = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        for idx in scenario_arrays['wind_nodes']:
            mask = (scenario_arrays['wind_nodes'] == idx)
            self.GWind[:,idx] = GWind[:,mask].sum(axis=1)

        self.flexible = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Spillage = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Charge = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Discharge = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Storage = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Deficit = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Import = np.zeros((self.intervals,self.lines), dtype=np.float64) 
        self.Export = np.zeros((self.intervals,self.lines), dtype=np.float64) 

    def _reliability(self, flexible, start=None, end=None):
        network = self.network
        networksteps = np.where(TRIANGULAR == network.shape[2])[0][0]
        
        Netload = (self.MLoad - self.GPV - self.GWind - self.GBaseload)[start:end]
        Netload -= flexible
        shape2d = intervals, nodes = len(Netload), self.nodes

        Pcapacity = self.CPHP * 1000
        Scapacity = self.CPHS * 1000

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

        self.flexible = flexible
        self.Spillage = Spillage
        self.Charge = Charge
        self.Discharge = Discharge
        self.Storage = Storage
        self.Deficit = Deficit
        self.Import = np.maximum(0, ImpExp)
        self.Export = -1 * np.minimum(0, ImpExp)

        self.TFlows = (Transmission).sum(axis=2)
        
        return Deficit

    def _calculate_costs(self):
        solution_cost = 0
        return solution_cost

    def _objective(self) -> List[float]:
        deficit = self._reliability(flexible=np.zeros((self.intervals, self.nodes), dtype=np.float64))      
        flexible = deficit.sum(axis=0) * self.resolution / self.years / (0.5 * (1 + self.efficiency))

        deficit = self._reliability(flexible=flexible)
        #pen_deficit = np.maximum(0., deficit.sum() * self.resolution - self.allowance) * 1000000 # MWh
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution) * 1000000 # MWh
        self.TFlowsAbs = np.abs(self.TFlows)

        cost = self._calculate_costs()

        loss = self.TFlowsAbs.sum(axis=0) * self.TLoss
        loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - loss)

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
        scenario_arrays['network'] = self.scenario.network.network  
        scenario_arrays['TLoss'] = np.array([self.scenario.lines[idx].loss_factor for idx in self.scenario.lines])
        #scenario_arrays['allowance']

        # Dynamic parameters
        scenario_arrays['TFlows'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.lines)+1))
        scenario_arrays['TFlowsAbs'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.lines)+1))
        scenario_arrays['Discharge'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)+1))
        scenario_arrays['Charge'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)+1))
        scenario_arrays['GFlexible'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)+1))              

        # Data Files
        solar_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)+1) if self.scenario.generators[idx].unit_type == 'solar']
        wind_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)+1) if self.scenario.generators[idx].unit_type == 'wind']
        baseload_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)+1) if self.scenario.generators[idx].unit_type == 'baseload']
        flexible_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)+1) if self.scenario.generators[idx].unit_type == 'flexible']
        scenario_arrays['TSPV'] = np.array([generator.data for generator in solar_generators]).T 
        scenario_arrays['TSWind'] = np.array([generator.data for generator in wind_generators]).T
        scenario_arrays['CBaseload'] = np.array([np.sum([generator.capacity for generator in baseload_generators if generator.node == self.scenario.nodes[idx].name]) for idx in range(0,max(self.scenario.nodes)+1)])
        scenario_arrays['CPeak'] = np.array([np.sum([generator.capacity for generator in flexible_generators if generator.node == self.scenario.nodes[idx].name]) for idx in range(0,max(self.scenario.nodes)+1)])

        # Decision variable indices
        scenario_arrays['pv_idx'] = scenario_arrays['TSPV'].shape[1]
        scenario_arrays['wind_idx'] = scenario_arrays['pv_idx'] + scenario_arrays['TSWind'].shape[1]
        scenario_arrays['storage_p_idx'] = scenario_arrays['wind_idx'] + len(self.scenario.storages)
        scenario_arrays['storage_e_idx'] = scenario_arrays['storage_p_idx'] + len(self.scenario.storages)
        scenario_arrays['lines_idx'] = scenario_arrays['storage_e_idx'] + len(self.scenario.lines)

        # Node ID lists
        scenario_arrays['pv_nodes'] = np.array([node_names[generator.node] for generator in solar_generators])
        scenario_arrays['wind_nodes'] = np.array([node_names[generator.node] for generator in wind_generators])
        scenario_arrays['storage_nodes'] = np.array([node_names[self.scenario.storages[idx].node] for idx in self.scenario.storages])
        
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