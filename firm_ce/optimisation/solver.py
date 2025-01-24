from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import differential_evolution

from firm_ce.network import get_transmission_flows_t
from firm_ce.constants import JIT_ENABLED
from firm_ce.components import calculate_costs

if JIT_ENABLED:
    from numba import njit, prange, float64, int64, boolean
    from numba.experimental import jitclass

    solution_spec = [
        ('x', float64[:]),
        ('evaluated', boolean),
        ('lcoe', float64),
        ('penalties', float64),

        ('MLoad', float64[:, :]),
        ('intervals', int64),
        ('nodes', int64),
        ('lines', int64),
        ('efficiency', float64),
        ('resolution', float64),
        ('years', int64),
        ('energy', float64),
        ('allowance', float64),

        ('solar_code', int64),
        ('wind_code', int64),
        ('flexible_code', int64),
        ('baseload_code', int64),

        ('generator_ids', int64[:]),
        ('generator_nodes', int64[:]),
        ('generator_capacities', float64[:]),
        ('generator_unit_types', int64[:]),
        ('generator_costs', float64[:, :]),

        ('TSPV', float64[:, :]),
        ('TSWind', float64[:, :]),
        ('CPeak', float64[:]),
        ('CBaseload', float64[:]),
        ('GBaseload', float64[:, :]),
        ('GFlexible', float64[:, :]),

        ('storage_ids', int64[:]),
        ('storage_nodes', int64[:]),
        ('storage_power_capacities', float64[:]),
        ('storage_energy_capacities', float64[:]),
        ('storage_costs', float64[:, :]),
        ('Discharge', float64[:, :]),
        ('Charge', float64[:, :]),

        ('line_ids', int64[:]),
        ('line_lengths', float64[:]),
        ('line_costs', float64[:, :]),
        
        ('network', int64[:, :, :, :]),
        ('networksteps', int64),

        ('TLoss', float64[:]),
        ('TFlows', float64[:, :]),
        ('TFlowsAbs', float64[:, :]),

        ('CPV', float64[:]),
        ('CWind', float64[:]),
        ('CPHP', float64[:]),
        ('CPHS', float64[:]),
        ('CTrans', float64[:]),

        ('solar_nodes', int64[:]),
        ('wind_nodes', int64[:]),
        ('flexible_nodes', int64[:]),
        ('baseload_nodes', int64[:]),

        ('CPHP_nodal', float64[:]),
        ('CPHS_nodal', float64[:]),
        ('GPV', float64[:, :]),
        ('GPV_nodal', float64[:, :]),
        ('GWind', float64[:, :]),
        ('GWind_nodal', float64[:, :]),

        ('CPeak_nodal', float64[:]),
        ('CBaseload_nodal', float64[:]),
        ('GBaseload_nodal', float64[:, :]),
        ('GFlexible_nodal', float64[:, :]),
        ('Spillage_nodal', float64[:, :]),
        ('Charge_nodal', float64[:, :]),
        ('Discharge_nodal', float64[:, :]),
        ('Storage_nodal', float64[:, :]),
        ('Deficit_nodal', float64[:, :]),
        ('Import_nodal', float64[:, :]),
        ('Export_nodal', float64[:, :]),

        ('GPV_annual', float64[:]),
        ('GWind_annual', float64[:]),
        ('GFlexible_annual', float64[:]),
        ('GBaseload_annual', float64[:]),
        ('GDischarge_annual', float64[:]),
        ('TFlowsAbs_annual', float64[:]),

        ('pv_cost_ids', int64[:]),
        ('wind_cost_ids', int64[:]),
        ('flexible_cost_ids', int64[:]),
        ('baseload_cost_ids', int64[:]),
    ]
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper
    
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    
    def prange(x):
        return range(x)
    
    solution_spec = []
    scenario_arrays_spec = []

@jitclass(solution_spec)
class Solution_SingleTime:
    def __init__(self, 
                x, 
                MLoad,
                TSPV,
                TSWind,
                network,
                intervals,
                nodes,
                lines,
                years,
                efficiency,
                resolution,
                allowance,
                generator_ids,
                generator_nodes,
                generator_capacities,
                generator_costs,
                generator_unit_types,
                storage_ids,
                storage_nodes,
                storage_power_capacities,
                storage_energy_capacities,
                storage_costs,
                line_ids,
                line_lengths,
                line_costs,
                TLoss,
                pv_idx,
                wind_idx,
                storage_p_idx,
                storage_e_idx,
                lines_idx,
                solar_nodes,
                wind_nodes,
                flexible_nodes,
                baseload_nodes,
                CPeak,
                CBaseload,
                pv_cost_ids,
                wind_cost_ids,
                flexible_cost_ids,
                baseload_cost_ids,
                networksteps,) -> None:

        self.x = x  
        self.evaluated=False   
        self.lcoe = 0.0
        self.penalties = 0.0

        self.MLoad = MLoad
        self.intervals = intervals
        self.nodes = nodes
        self.lines = lines
        self.efficiency = efficiency
        self.resolution = resolution
        self.years = years
        self.energy = self.MLoad.sum() * self.resolution / self.years
        self.allowance = allowance

        # Unit Types
        self.solar_code = 0
        self.wind_code = 1
        self.flexible_code = 2
        self.baseload_code = 3

        # Generators
        self.generator_ids = generator_ids
        self.generator_nodes = generator_nodes
        self.generator_capacities = generator_capacities
        self.generator_costs = generator_costs
        self.generator_unit_types = generator_unit_types

        self.TSPV = TSPV 
        self.TSWind = TSWind 
        self.CPeak = CPeak
        self.CBaseload = CBaseload
        self.GBaseload = self.CBaseload * np.ones((self.intervals, len(self.CBaseload)), dtype=np.float64)
        self.GFlexible = np.zeros((self.intervals, len(self.CPeak)), dtype=np.float64)

        # Storages
        self.storage_ids = storage_ids
        self.storage_nodes = storage_nodes
        self.storage_power_capacities = storage_power_capacities
        self.storage_energy_capacities = storage_energy_capacities
        self.storage_costs = storage_costs

        self.Discharge = np.zeros((intervals, len(storage_ids)), dtype=np.float64)
        self.Charge = np.zeros((intervals, len(storage_ids)), dtype=np.float64)

        # Lines
        self.line_ids = line_ids
        self.line_lengths = line_lengths
        self.line_costs = line_costs
        
        self.network = network
        self.networksteps = networksteps
        self.TLoss = TLoss
        self.TFlows = np.zeros((intervals, lines), dtype=np.float64)
        self.TFlowsAbs = np.zeros((intervals, lines), dtype=np.float64)

        # Decision Variables
        self.CPV = x[: pv_idx]
        self.CWind = x[pv_idx : wind_idx]
        self.CPHP = x[wind_idx : storage_p_idx]
        self.CPHS = x[storage_p_idx : storage_e_idx]
        self.CTrans = x[storage_e_idx :] ###### THIS NEEDS TO BE TREATED AS A BOUND

        # Nodal Values
        self.solar_nodes = solar_nodes
        self.wind_nodes = wind_nodes
        self.flexible_nodes = flexible_nodes 
        self.baseload_nodes = baseload_nodes
        self.storage_nodes = self.storage_nodes

        self.CPHP_nodal = self._fill_nodal_array_1d(self.CPHP, self.storage_nodes)
        self.CPHS_nodal = self._fill_nodal_array_1d(self.CPHS, self.storage_nodes)   
        self.GPV = self.CPV[np.newaxis, :] * TSPV  * 1000
        self.GPV_nodal = self._fill_nodal_array_2d(self.GPV, self.solar_nodes)
        self.GWind = self.CWind[np.newaxis, :] * TSWind * 1000 
        self.GWind_nodal = self._fill_nodal_array_2d(self.GWind, self.wind_nodes)
        self.CPeak_nodal = self._fill_nodal_array_1d(self.CPeak, self.flexible_nodes)
        self.CBaseload_nodal = self._fill_nodal_array_1d(self.CBaseload, self.baseload_nodes)
        self.GBaseload_nodal = self._fill_nodal_array_2d(self.GBaseload, self.baseload_nodes)

        #print("Nodals: ", self.CPHP_nodal, self.CPHS_nodal)

        self.GFlexible_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Spillage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Charge_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Discharge_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Storage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Deficit_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Import_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64) 
        self.Export_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64)   

        self.GPV_annual = np.zeros(0, dtype=np.float64)
        self.GWind_annual = np.zeros(0, dtype=np.float64)
        self.GFlexible_annual = np.zeros(0, dtype=np.float64)
        self.GBaseload_annual = np.zeros(0, dtype=np.float64)
        self.GDischarge_annual = np.zeros(0, dtype=np.float64)
        self.TFlowsAbs_annual = np.zeros(0, dtype=np.float64)

        # Cost Values
        self.pv_cost_ids = pv_cost_ids
        self.wind_cost_ids = wind_cost_ids
        self.flexible_cost_ids = flexible_cost_ids
        self.baseload_cost_ids = baseload_cost_ids

    def _fill_nodal_array_2d(self, generation_array, node_array):
        result = np.zeros((self.intervals, self.nodes), dtype=np.float64)

        for t in range(self.intervals):
            for n in range(len(node_array)):
                node_idx = node_array[n]
                result[t, node_idx] += generation_array[t, n]

        return result
    
    def _fill_nodal_array_1d(self, capacity_array, node_array):
        result = np.zeros(self.nodes, dtype=np.float64)

        for n in range(len(node_array)):
            node_idx = node_array[n]
            result[node_idx] += capacity_array[n]

        return result

    def _reliability(self, flexible, start=None, end=None):
        network = self.network
        
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
                #print("1: ", Deficitt)
                Transmissiont = get_transmission_flows_t(Deficitt, Surplust, Hcapacity, network, self.networksteps, 
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
                #print("2: ", Netloadt, Discharget, Pcapacity, Scapacity, Storaget_1, Charget)
                Transmissiont = get_transmission_flows_t(Fillt, Surplust, Hcapacity, network, self.networksteps,
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
    
    def _apportion_nodal_array(self, capacity_array, nodal_generation, node_array):
        n_components = node_array.shape[0]

        cap_node_sum = np.zeros(self.nodes, dtype=np.float64)
        for i in range(n_components):
            node_i = node_array[i]
            cap_node_sum[node_i] += capacity_array[i]

        result = np.zeros((self.intervals, n_components), dtype=np.float64)
        for i in range(n_components):
            node_i = node_array[i]
            denom = cap_node_sum[node_i]
            if denom > 0.0:
                ratio = capacity_array[i] / denom
                for t in range(self.intervals):
                    result[t, i] = ratio * nodal_generation[t, node_i]
            else:
                # If the node's total capacity is zero, fall back to "no scaling"
                for t in range(self.intervals):
                    result[t, i] = nodal_generation[t, node_i]

        return result
    
    def _apportion_nodal_generation(self):
        self.GFlexible = self._apportion_nodal_array(self.CPeak,self.GFlexible_nodal,self.flexible_nodes)
        
        self.GBaseload = self._apportion_nodal_array(self.CBaseload,self.GBaseload_nodal,self.baseload_nodes)

        self.Discharge = self._apportion_nodal_array(self.CPHP,self.Discharge_nodal,self.storage_nodes)

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
        print('Success 1')

        # Unit types
        unit_types = {'solar': 0,
                                        'wind': 1,
                                        'flexible': 2,
                                        'baseload': 3
                                        } ############ GENERATE THIS IN SCENARIOS
        
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
            [unit_types[self.scenario.generators[idx].unit_type]
            for idx in self.scenario.generators],
            dtype=np.int64
        )

        scenario_arrays['generator_costs'] = np.zeros(
            (8, len(self.scenario.generators)), dtype=np.float64
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

        print('Success 2')

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

        scenario_arrays['storage_costs'] = np.zeros(
            (7, len(self.scenario.storages)), dtype=np.float64
        )

        scenario_arrays['Discharge'] = np.zeros(
            (scenario_arrays['intervals'], len(self.scenario.storages)), dtype=np.float64
        )
        scenario_arrays['Charge'] = np.zeros(
            (scenario_arrays['intervals'], len(self.scenario.storages)), dtype=np.float64
        )

        print('Success 3')

        # Lines
        scenario_arrays['line_ids'] = np.array(
            [self.scenario.lines[idx].id for idx in self.scenario.lines],
            dtype=np.int64
        )

        """ scenario_arrays.line_capacities = np.array(
            [self.scenario.lines[idx].capacity for idx in self.scenario.lines],
            dtype=np.float64
        ) """

        scenario_arrays['line_lengths'] = np.array(
            [self.scenario.lines[idx].length for idx in self.scenario.lines],
            dtype=np.float64
        )

        scenario_arrays['line_costs'] = np.zeros(
            (7, len(self.scenario.lines)), dtype=np.float64
        )
        print('Success 4',type(self.scenario.network.network))
        scenario_arrays['network'] = self.scenario.network.network
        scenario_arrays['networksteps'] = self.scenario.network.networksteps
        print('Success 4.1')
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
        scenario_arrays['storage_p_idx'] = scenario_arrays['wind_idx'] + len(self.scenario.storages)
        scenario_arrays['storage_e_idx'] = scenario_arrays['storage_p_idx'] + len(self.scenario.storages)
        scenario_arrays['lines_idx'] = scenario_arrays['storage_e_idx'] + len(self.scenario.lines)

        print('Success 5')

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

        print('Success 6')

        solar_code = 0
        wind_code = 1
        flexible_code = 2
        baseload_code = 3

        scenario_arrays['pv_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == solar_code)]
        scenario_arrays['wind_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == wind_code)]
        scenario_arrays['flexible_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == flexible_code)]
        scenario_arrays['baseload_cost_ids'] = scenario_arrays['generator_ids'][np.where(scenario_arrays['generator_unit_types'] == baseload_code)]

        scenario_arrays['solar_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == solar_code)[0]] 
        scenario_arrays['wind_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == wind_code)[0]] 
        scenario_arrays['flexible_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == flexible_code)[0]] 
        scenario_arrays['baseload_nodes'] = scenario_arrays['generator_nodes'][np.where(scenario_arrays['generator_unit_types'] == baseload_code)[0]] 
        scenario_arrays['CPeak'] = scenario_arrays['generator_capacities'][np.where(scenario_arrays['generator_unit_types'] == flexible_code)[0]]
        scenario_arrays['CBaseload'] = scenario_arrays['generator_capacities'][np.where(scenario_arrays['generator_unit_types'] == baseload_code)[0]]
        
        return scenario_arrays

    def _single_time(self):
        scenario_arrays = self._prepare_scenario_arrays()

        MLoad = scenario_arrays["MLoad"] 
        TSPV = scenario_arrays["TSPV"]   
        TSWind = scenario_arrays["TSWind"]
        network = scenario_arrays["network"]
        networksteps = scenario_arrays['networksteps'] 

        intervals = scenario_arrays["intervals"]
        nodes = scenario_arrays["nodes"]
        lines = scenario_arrays["lines"]
        years = scenario_arrays["years"]
        efficiency = scenario_arrays["efficiency"]
        resolution = scenario_arrays["resolution"]
        allowance = scenario_arrays["allowance"]

        generator_ids = scenario_arrays["generator_ids"]
        generator_nodes = scenario_arrays["generator_nodes"]
        generator_capacities = scenario_arrays["generator_capacities"]
        generator_costs = scenario_arrays["generator_costs"]
        generator_unit_types = scenario_arrays["generator_unit_types"]

        storage_ids = scenario_arrays["storage_ids"]
        storage_nodes = scenario_arrays["storage_nodes"]
        storage_power_capacities = scenario_arrays["storage_power_capacities"]
        storage_energy_capacities = scenario_arrays["storage_energy_capacities"]
        storage_costs = scenario_arrays["storage_costs"]

        line_ids = scenario_arrays["line_ids"]
        line_lengths = scenario_arrays["line_lengths"]
        line_costs = scenario_arrays["line_costs"]
        TLoss = scenario_arrays["TLoss"]

        pv_idx = scenario_arrays["pv_idx"]
        wind_idx = scenario_arrays["wind_idx"]
        storage_p_idx = scenario_arrays["storage_p_idx"]
        storage_e_idx = scenario_arrays["storage_e_idx"]
        lines_idx = scenario_arrays["lines_idx"]

        solar_nodes = scenario_arrays['solar_nodes'] 
        wind_nodes = scenario_arrays['wind_nodes'] 
        flexible_nodes = scenario_arrays['flexible_nodes'] 
        baseload_nodes = scenario_arrays['baseload_nodes'] 
        CPeak = scenario_arrays['CPeak'] 
        CBaseload = scenario_arrays['CBaseload']

        pv_cost_ids = scenario_arrays['pv_cost_ids']
        wind_cost_ids = scenario_arrays['wind_cost_ids']
        flexible_cost_ids = scenario_arrays['flexible_cost_ids']
        baseload_cost_ids = scenario_arrays['baseload_cost_ids']

        self.result = differential_evolution(
            x0=self.decision_x0,
            func=parallel_wrapper, 
            bounds=list(zip(self.lower_bounds, self.upper_bounds)), 
            args=(MLoad,
                TSPV,
                TSWind,
                network,
                intervals,
                nodes,
                lines,
                years,
                efficiency,
                resolution,
                allowance,
                generator_ids,
                generator_nodes,
                generator_capacities,
                generator_costs,
                generator_unit_types,
                storage_ids,
                storage_nodes,
                storage_power_capacities,
                storage_energy_capacities,
                storage_costs,
                line_ids,
                line_lengths,
                line_costs,
                TLoss,
                pv_idx,
                wind_idx,
                storage_p_idx,
                storage_e_idx,
                lines_idx,
                solar_nodes,
                wind_nodes,
                flexible_nodes,
                baseload_nodes,
                CPeak,
                CBaseload,
                pv_cost_ids,
                wind_cost_ids,
                flexible_cost_ids,
                baseload_cost_ids,
                networksteps,),
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

@njit(parallel=True)
def parallel_wrapper(xs,
                    MLoad,
                    TSPV,
                    TSWind,
                    network,
                    intervals,
                    nodes,
                    lines,
                    years,
                    efficiency,
                    resolution,
                    allowance,
                    generator_ids,
                    generator_nodes,
                    generator_capacities,
                    generator_costs,
                    generator_unit_types,
                    storage_ids,
                    storage_nodes,
                    storage_power_capacities,
                    storage_energy_capacities,
                    storage_costs,
                    line_ids,
                    line_lengths,
                    line_costs,
                    TLoss,
                    pv_idx,
                    wind_idx,
                    storage_p_idx,
                    storage_e_idx,
                    lines_idx,
                    solar_nodes,
                    wind_nodes,
                    flexible_nodes,
                    baseload_nodes,
                    CPeak,
                    CBaseload,
                    pv_cost_ids,
                    wind_cost_ids,
                    flexible_cost_ids,
                    baseload_cost_ids,
                    networksteps,):
    result = np.empty(xs.shape[1], dtype=np.float64)
    for i in prange(xs.shape[1]):
        result[i] = objective_st(xs[:,i], 
                                MLoad,
                                TSPV,
                                TSWind,
                                network,
                                intervals,
                                nodes,
                                lines,
                                years,
                                efficiency,
                                resolution,
                                allowance,
                                generator_ids,
                                generator_nodes,
                                generator_capacities,
                                generator_costs,
                                generator_unit_types,
                                storage_ids,
                                storage_nodes,
                                storage_power_capacities,
                                storage_energy_capacities,
                                storage_costs,
                                line_ids,
                                line_lengths,
                                line_costs,
                                TLoss,
                                pv_idx,
                                wind_idx,
                                storage_p_idx,
                                storage_e_idx,
                                lines_idx,
                                solar_nodes,
                                wind_nodes,
                                flexible_nodes,
                                baseload_nodes,
                                CPeak,
                                CBaseload,
                                pv_cost_ids,
                                wind_cost_ids,
                                flexible_cost_ids,
                                baseload_cost_ids,
                                networksteps,)
    return result

@njit
def objective_st(x, 
                MLoad,
                TSPV,
                TSWind,
                network,
                intervals,
                nodes,
                lines,
                years,
                efficiency,
                resolution,
                allowance,
                generator_ids,
                generator_nodes,
                generator_capacities,
                generator_costs,
                generator_unit_types,
                storage_ids,
                storage_nodes,
                storage_power_capacities,
                storage_energy_capacities,
                storage_costs,
                line_ids,
                line_lengths,
                line_costs,
                TLoss,
                pv_idx,
                wind_idx,
                storage_p_idx,
                storage_e_idx,
                lines_idx,
                solar_nodes,
                wind_nodes,
                flexible_nodes,
                baseload_nodes,
                CPeak,
                CBaseload,
                pv_cost_ids,
                wind_cost_ids,
                flexible_cost_ids,
                baseload_cost_ids,
                networksteps,):
    solution = Solution_SingleTime(x,
                                MLoad,
                                TSPV,
                                TSWind,
                                network,
                                intervals,
                                nodes,
                                lines,
                                years,
                                efficiency,
                                resolution,
                                allowance,
                                generator_ids,
                                generator_nodes,
                                generator_capacities,
                                generator_costs,
                                generator_unit_types,
                                storage_ids,
                                storage_nodes,
                                storage_power_capacities,
                                storage_energy_capacities,
                                storage_costs,
                                line_ids,
                                line_lengths,
                                line_costs,
                                TLoss,
                                pv_idx,
                                wind_idx,
                                storage_p_idx,
                                storage_e_idx,
                                lines_idx,
                                solar_nodes,
                                wind_nodes,
                                flexible_nodes,
                                baseload_nodes,
                                CPeak,
                                CBaseload,
                                pv_cost_ids,
                                wind_cost_ids,
                                flexible_cost_ids,
                                baseload_cost_ids,
                                networksteps,)
    solution.evaluate()
    #print(solution.lcoe + solution.penalties)
    return solution.lcoe + solution.penalties

