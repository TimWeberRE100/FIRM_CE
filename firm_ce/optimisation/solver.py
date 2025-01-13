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
        #self.allowance = scenario_arrays['allowance']

        self.TFlows = scenario_arrays['TFlows']
        self.TFlowsAbs = scenario_arrays['TFlowsAbs']
        self.Discharge = scenario_arrays['Discharge']
        self.Charge = scenario_arrays['Charge']
        self.GFlexible = scenario_arrays['GFlexible']
        self.GBaseload = scenario_arrays['GBaseload']
        self.network = scenario_arrays['network']

        self.GPV = scenario_arrays['GPV']
        self.GWind = scenario_arrays['GWind']
        self.CPHP = scenario_arrays['CPHP']
        self.CPHS = scenario_arrays['CPHS']
        self.CTrans = scenario_arrays['CTrans']

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
        trans_tdc_mask = self.trans_tdc_mask
        networksteps = np.where(TRIANGULAR == network.shape[2])[0][0]
        
        Netload = (self.MLoad - self.GPV - self.GWind - self.baseload)[start:end]
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

        self.CTrans = (np.atleast_3d(trans_tdc_mask).T*Transmission).sum(axis=2)
        
        return Deficit

    def _calculate_costs(self):
        solution_cost = SolutionCost(self.generators, self.storages, self.lines)
        return solution_cost.cost

    def _objective(self) -> List[float]:
        deficit = self._reliability(flexible=np.zeros((self.intervals, self.nodes), dtype=np.float64))      
        flexible = deficit.sum(axis=0) * self.resolution / self.years / (0.5 * (1 + self.efficiency))

        deficit = self._reliability(flexible=flexible)
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution - self.allowance) * 1000000 # MWh
        self.TFlowsAbs = np.abs(self.TFlows)

        cost, _ = self._calculate_costs()

        loss = np.zeros(len(self.network_mask), dtype=np.float64)
        loss[self.network_mask] = self.TFlowsAbs.sum(axis=0) * self.TLoss[self.network_mask]
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
        self.upper_bounds = None
        self.lower_bounds = None   
        self.result = None

    def _prepare_scenario_arrays(self):
        
        scenario_arrays = {}

        # Static parameters
        scenario_arrays['MLoad'] = np.array([self.scenario.nodes[idx].demand_data for idx in range(0,max(self.scenario.nodes))], dtype=np.float64)
        scenario_arrays['intervals'], scenario_arrays['nodes'] = scenario_arrays['MLoad'].shape
        scenario_arrays['lines'] = len(self.scenario.lines)
        scenario_arrays['years'] = self.scenario.final_year - self.scenario.first_year   
        scenario_arrays['efficiency'] = 0.8
        scenario_arrays['resolution'] = self.scenario.resolution
        scenario_arrays['network'] = self.scenario.network.network  
        #scenario_arrays['allowance']

        # Dynamic parameters
        scenario_arrays['TFlows'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.lines)))
        scenario_arrays['TFlowsAbs'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.lines)))
        scenario_arrays['Discharge'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)))
        scenario_arrays['Charge'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)))
        scenario_arrays['GFlexible'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)))              

        # Data Files
        solar_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)) if self.scenario.generators[idx].unit_type == 'solar']
        wind_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)) if self.scenario.generators[idx].unit_type == 'wind']
        baseload_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)) if self.scenario.generators[idx].unit_type == 'baseload']
        flexible_generators = [self.scenario.generators[idx] for idx in range(0,max(self.scenario.generators)) if self.scenario.generators[idx].unit_type == 'flexible']
        scenario_arrays['TSPV'] = np.array([np.sum([generator.data for generator in solar_generators if generator.node == self.scenario.nodes[idx].name], axis=1) for idx in range(0,max(self.scenario.nodes))])
        scenario_arrays['TSWind'] = np.array([np.sum([generator.data for generator in wind_generators if generator.node == self.scenario.nodes[idx].name], axis=1) for idx in range(0,max(self.scenario.nodes))])
        scenario_arrays['GBaseload'] = np.array([np.sum([generator.data for generator in baseload_generators if generator.node == self.scenario.nodes[idx].name], axis=1) for idx in range(0,max(self.scenario.nodes))])
        scenario_arrays['CPeak'] = np.array([np.sum([generator.capacity for generator in flexible_generators if generator.node == self.scenario.nodes[idx].name], axis=1) for idx in range(0,max(self.scenario.nodes))])

        # Decision Variables
        scenario_arrays['CPV'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes))) 
        scenario_arrays['CWind'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)))
        scenario_arrays['CPHP'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)))
        scenario_arrays['CPHS'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.nodes)))
        scenario_arrays['CTrans'] = np.zeros((scenario_arrays['intervals'],max(self.scenario.lines)))
        
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
    @njit(parallel=True)
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