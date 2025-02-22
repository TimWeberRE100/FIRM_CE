import numpy as np
from typing import List

from firm_ce.network import get_transmission_flows_t
from firm_ce.constants import JIT_ENABLED, EPSILON_FLOAT64, NP_FLOAT_MAX, NP_FLOAT_MIN
from firm_ce.components.costs import calculate_costs, annualisation_component
import firm_ce.network.frequency as frequency
import firm_ce.helpers as helpers
import firm_ce.network.cwt as cwt

if JIT_ENABLED:
    from numba import float64, int64, boolean, njit, prange
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
        ('storage_unit_types', int64[:]),


        ('flexible_ids', int64[:]),
        ('nodes_with_balancing', int64[:]),
        ('max_frequency', float64),
        ('storage_durations', float64[:]),
        ('storage_costs', float64[:, :]),
        ('Discharge', float64[:, :]),
        ('Charge', float64[:, :]),
        ('storage_d_efficiencies', float64[:]),
        ('storage_c_efficiencies', float64[:]),

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
        ('CFlexible', float64[:]),
        ('CPHP', float64[:]),
        ('CPHS', float64[:]),
        ('CTrans', float64[:]),
        ('balancing_W_x', float64[:]),

        ('solar_nodes', int64[:]),
        ('wind_nodes', int64[:]),
        ('flexible_nodes', int64[:]),
        ('baseload_nodes', int64[:]),

        ('CFlexible_nodal', float64[:]),
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
        ('NetBalancing_nodal', float64[:, :]),
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
        ('storage_cost_ids', int64[:]),
        ('line_cost_ids', int64[:]),

        ('storage_p_profiles', float64[:,:]),
        ('storage_e_profiles', float64[:,:]),
        ('storage_p_capacities', float64[:]),
        ('storage_e_capacities', float64[:]),
        ('flexible_p_profiles', float64[:,:]),
        ('flexible_p_capacities', float64[:]),
        ('balancing_p_profiles_ifft', float64[:,:,:]),
        ('storage_p_capacities_ifft', float64[:,:]),
        ('balancing_W_x_nodal', float64[:,:]),
        ('balancing_W_cutoffs', float64[:,:]),
        ('balancing_W_thresholds', float64[:,:]),
        ('nodal_balancing_count', int64[:]),

        ('nodes_with_balancing', int64[:]),
        ('balancing_ids', int64[:]),
        ('balancing_nodes', int64[:]),
        ('balancing_order', int64[:]),
        ('balancing_storage_tag', int64[:]),
        ('balancing_d_efficiencies', float64[:]),
        ('balancing_c_efficiencies', float64[:]),
        ('balancing_c_constraint', float64[:]),
        ('balancing_d_constraint', float64[:]),
        ('balancing_e_constraints', float64[:]),
        ('balancing_costs', float64[:,:]),
    ]
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper
    
    def prange(x):
        return range(x)
    
    solution_spec = []

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
                storage_unit_types,
                flexible_ids,
                nodes_with_balancing,
                max_frequency,
                storage_durations,
                storage_costs,
                line_ids,
                line_lengths,
                line_costs,
                TLoss,
                pv_idx,
                wind_idx,
                flexible_p_idx,
                storage_p_idx,
                storage_e_idx,
                lines_idx,
                balancing_W_idx,
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
                storage_cost_ids,
                line_cost_ids,
                networksteps,
                storage_d_efficiencies,
                storage_c_efficiencies,) -> None:

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
        self.storage_unit_types = storage_unit_types
        self.max_frequency = max_frequency
        self.storage_durations = storage_durations
        self.storage_costs = storage_costs
        self.storage_d_efficiencies = storage_d_efficiencies
        self.storage_c_efficiencies = storage_c_efficiencies

        self.Discharge = np.zeros((intervals, len(storage_ids)), dtype=np.float64)
        self.Charge = np.zeros((intervals, len(storage_ids)), dtype=np.float64)
        self.Storage = np.zeros((intervals, len(storage_ids)), dtype=np.float64)

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
        self.CFlexible = x[wind_idx : flexible_p_idx]
        self.CPHP = x[flexible_p_idx : storage_p_idx]
        self.CPHS = x[storage_p_idx : storage_e_idx]
        self.CTrans = x[storage_e_idx : lines_idx]
        self.balancing_W_x = x[lines_idx : ]

        """ print(self.CPV,self.CWind,self.CFlexible,self.CPHS,self.CTrans,self.balancing_W_x) """

        # Flexible
        self.flexible_ids = flexible_ids
        self.nodes_with_balancing = nodes_with_balancing
        self.balancing_ids = np.hstack((storage_ids, flexible_ids))
        self.balancing_nodes = np.hstack((storage_nodes, flexible_nodes))
        self.balancing_order = np.arange(len(self.balancing_ids), dtype=np.int64)
        self.balancing_storage_tag = np.hstack(
                                        (np.full(len(storage_ids), True, dtype=np.bool_),
                                        np.full(len(flexible_ids), False, dtype=np.bool_))
                                    )
        self.balancing_flexible_tag = np.hstack(
                                        (np.full(len(storage_ids), False, dtype=np.bool_),
                                        np.full(len(flexible_ids), True, dtype=np.bool_))
                                    )
        self.balancing_d_efficiencies = np.hstack(
                                                (storage_d_efficiencies, 
                                                np.ones(len(flexible_ids), dtype=np.float64))
                                                )
        self.balancing_c_efficiencies = np.hstack(
                                                    (storage_c_efficiencies, 
                                                    np.ones(len(flexible_ids), dtype=np.float64))
                                                )
        self.balancing_c_constraint = np.hstack(
                                        (-1 * self.CPHP, 
                                        np.zeros(len(flexible_ids), dtype=np.float64))
                                      )
        self.balancing_d_constraint = np.hstack(
                                        (self.CPHP, 
                                        self.CFlexible)
                                      )
        self.balancing_e_constraints = np.hstack(
                                        (self.CPHS, 
                                        np.full(len(flexible_ids), NP_FLOAT_MAX / 10000, dtype=np.float64)) # Divide by 10000 to prevent overflow when converting GWh to MWh
                                      )
        
        flexible_mask = helpers.isin_numba(np.arange(self.generator_costs.shape[1], dtype=np.int64), flexible_cost_ids)
        #print(flexible_mask, self.storage_costs.shape,self.generator_costs[:, flexible_mask].shape)
        self.balancing_costs = np.hstack(
                                        (self.storage_costs, 
                                        self.generator_costs[:, flexible_mask])
                                        )

        """ nodal_durations = 
        for idx in range(len(storage_durations)):
            if storage_durations[idx] > 0:
                self.CPHS[idx] = self.CPHP[idx] * storage_durations[idx] """

        # Nodal Values
        self.solar_nodes = solar_nodes
        self.wind_nodes = wind_nodes
        self.flexible_nodes = flexible_nodes 
        self.baseload_nodes = baseload_nodes
        self.storage_nodes = self.storage_nodes

        self.CFlexible_nodal = self._fill_nodal_array_1d(self.CFlexible, self.flexible_nodes)
        self.CPHP_nodal = self._fill_nodal_array_1d(self.CPHP, self.storage_nodes)
        self.CPHS_nodal = self._fill_nodal_array_1d(self.CPHS, self.storage_nodes)    
        self.GPV = self.CPV[np.newaxis, :] * TSPV  * 1000
        self.GPV_nodal = self._fill_nodal_array_2d(self.GPV, self.solar_nodes)
        self.GWind = self.CWind[np.newaxis, :] * TSWind * 1000 
        self.GWind_nodal = self._fill_nodal_array_2d(self.GWind, self.wind_nodes)
        self.CPeak_nodal = self._fill_nodal_array_1d(self.CPeak, self.flexible_nodes)
        self.CBaseload_nodal = self._fill_nodal_array_1d(self.CBaseload, self.baseload_nodes)
        self.GBaseload_nodal = self._fill_nodal_array_2d(self.GBaseload, self.baseload_nodes)

        self.GFlexible_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Spillage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Charge_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Discharge_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.NetBalancing_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Storage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Deficit_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Import_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64) 
        self.Export_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64)   

        self.nodal_balancing_count = np.zeros(self.nodes, dtype=np.int64)
        for node_idx in range(len(self.nodal_balancing_count)):
            for i in self.balancing_nodes:
                if i == node_idx:
                    self.nodal_balancing_count[node_idx]+=1

        self.balancing_W_x_nodal = -1 * np.ones((self.nodes, max(self.nodal_balancing_count)), dtype=np.float64)

        for node_idx in range(len(self.nodal_balancing_count)):
            for current_node_count in range(self.nodal_balancing_count[node_idx]-1):
                W_idx = sum(self.nodal_balancing_count[:node_idx]) - len(self.nodal_balancing_count[:node_idx]) + current_node_count if node_idx > 0 else current_node_count
                self.balancing_W_x_nodal[node_idx, current_node_count] = self.balancing_W_x[W_idx]

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
        self.storage_cost_ids = storage_cost_ids
        self.line_cost_ids = line_cost_ids

        # Frequency attributes
        self.balancing_p_profiles_ifft = np.zeros((self.intervals,self.nodes,max(self.nodal_balancing_count)), dtype=np.float64)
        self.balancing_W_cutoffs = np.zeros((self.nodes, self.balancing_W_x_nodal.shape[1] + 2), dtype=np.float64)   

        self.storage_p_profiles = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)
        self.storage_e_profiles = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)                
        self.storage_e_capacities = np.zeros(len(storage_ids), dtype=np.float64)
        self.storage_p_capacities = np.zeros(len(storage_ids), dtype=np.float64)
        self.flexible_p_profiles = np.zeros((self.intervals,len(flexible_ids)), dtype=np.float64)
        self.flexible_p_capacities = np.zeros(len(flexible_ids), dtype=np.float64)

        #print(x)      

        for node_idx in self.storage_nodes:
            for W_idx in range(self.balancing_W_x_nodal.shape[1]):
                if self.balancing_W_x_nodal[node_idx,W_idx] > -0.5:
                    self.balancing_W_cutoffs[node_idx,W_idx+1] = self.balancing_W_x_nodal[node_idx,W_idx]
                else:
                    self.balancing_W_cutoffs[node_idx,W_idx+1] = max_frequency
            self.balancing_W_cutoffs[node_idx,-1] = max_frequency   

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

    def _reliability(self, start=None, end=None):
        network = self.network
        
        Netload = (self.MLoad - self.GPV - self.GWind - self.GBaseload_nodal)[start:end]

        balancing_p_profiles_ifft = self._filter_balancing_profiles(Netload)
        balancing_p_profiles, flexible_p_profiles, deficit, surplus = self._determine_constrained_balancing(balancing_p_profiles_ifft)

        #Netload = deficit+surplus

        SPower = balancing_p_profiles
        Storage = np.zeros(SPower.shape, dtype=np.float64)
        storage_d_efficiencies = self.balancing_d_efficiencies[self.balancing_storage_tag]
        storage_c_efficiencies = self.balancing_d_efficiencies[self.balancing_storage_tag]
        storage_order = self.balancing_order[self.balancing_storage_tag]
        SPower_nodal = self._fill_nodal_array_2d(SPower, self.storage_nodes)
     
        FPower = flexible_p_profiles
        FPower_nodal = self._fill_nodal_array_2d(FPower, self.flexible_nodes)
        flexible_order = self.balancing_order[self.balancing_flexible_tag]
        F_variable_costs = self.balancing_costs[3,flexible_order] ##### ADD FUEL COSTS

        shape2d = intervals, nodes = len(Netload), self.nodes

        S_p_capacity_nodal = self.CPHP_nodal * 1000
        S_p_capacity = self.CPHP * 1000
        S_e_capacity = self.CPHS * 1000
        Fcapacity_nodal = self.CFlexible_nodal * 1000
        Fcapacity = self.CFlexible * 1000

        Hcapacity = self.CTrans * 1000 # GW to MW
        ntrans = len(self.CTrans)
        
        Deficit = np.zeros(shape2d, dtype=np.float64)
        Discharge = np.zeros(shape2d, dtype=np.float64)
        Charge = np.zeros(shape2d, dtype=np.float64)
        Transmission = np.zeros((intervals, ntrans, nodes), dtype = np.float64)
        Storaget_1 = 0.5 * self.CPHS * 1000

        for t in range(intervals):
            Netloadt = Netload[t]

            Discharget_original = np.maximum(SPower_nodal[t,:], 0)
            Charget_original = -1 * np.minimum(SPower_nodal[t,:], 0)

            Storage_p_lb_t = Storaget_1 / self.resolution / storage_d_efficiencies
            Storage_p_lb_t_nodal = self._fill_nodal_array_1d(Storage_p_lb_t, self.storage_nodes)
            Discharget = np.minimum(Discharget_original, Storage_p_lb_t_nodal)
          
            FPowert = FPower_nodal[t,:] ####### ADD ENERGY CONSTRAINTS?

            Deficitt = np.maximum(Netloadt - Discharget,0)            
            Transmissiont=np.zeros((ntrans, nodes), dtype=np.float64)
        
            if Deficitt.sum() > 1e-6:
                # Fill deficits with transmission allowing drawing down from neighbours storage reserves
                
                Surplust = -1 * np.minimum(0, Netloadt) +  (Fcapacity_nodal - FPowert) + (np.minimum(S_p_capacity, Storage_p_lb_t_nodal) - Discharget) 

                Transmissiont = get_transmission_flows_t(Deficitt, Surplust, Hcapacity, network, self.networksteps, 
                                    np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))
                
                Netloadt = Netload[t] - Transmissiont.sum(axis=0)
                
                Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), S_p_capacity), Storage_p_lb_t_nodal) 
                FPowert = np.minimum(Netloadt - Discharget, Fcapacity_nodal)
                
            Storage_p_ub_t = (S_e_capacity - Storaget_1) / self.resolution / storage_c_efficiencies
            Storage_p_ub_t_nodal = self._fill_nodal_array_1d(Storage_p_ub_t, self.storage_nodes)
            Charget = np.minimum(Charget_original, Storage_p_ub_t_nodal)  
            Surplust = -1 * np.minimum(0, Netloadt + Charget)# charge itself first, then distribute

            if Surplust.sum() > 1e-6:
                # raise KeyboardInterrupt
                # Distribute surplus energy with transmission to areas with spare charging capacity
                Fillt = (Discharget + FPowert # remaining load not met by non-flexible gen and transmission
                        + np.minimum(Storage_p_ub_t_nodal, S_p_capacity) #full charging capacity
                        - Charget) #charge capacity already in use
                
                Transmissiont = get_transmission_flows_t(Fillt, Surplust, Hcapacity, network, self.networksteps,
                                    np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))

                Netloadt = Netload[t] - Transmissiont.sum(axis=0)
                Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), S_p_capacity_nodal), Storage_p_ub_t_nodal)
                Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), S_p_capacity_nodal), Storage_p_lb_t_nodal)
                FPowert = np.minimum(Netloadt - Discharget, Fcapacity_nodal)

            # Apportion to individual storages/flexible
            Storaget = np.zeros(Storaget_1.shape, dtype=np.float64)
            Discharget_update = Discharget - Discharget_original
            Charget_update = Charget - Charget_original
            FPowert_update = FPowert - FPower_nodal[t,:]
            ############## STILL NEED TO FIX APPORTIONING
            for node in range(self.nodes):
                # Storages
                storage_mask = self.storage_nodes == node

                Storaget_1_node = Storaget_1[storage_mask]
                storage_order_node = storage_order[storage_mask]
                sorted_storage_indices = np.argsort(Storaget_1_node)

                Discharget_node = Discharget[node]
                Charget_node = Charget[node]

                for i in range(len(sorted_storage_indices)):
                    reverse_i = len(sorted_storage_indices) - sorted_storage_indices[i] - 1
                    storage_order_i = storage_order_node[reverse_i]

                    Discharget_update[storage_order_i] = np.minimum(np.minimum(Discharget_node, S_p_capacity[storage_order_i] - SPower[t,storage_order_i]),
                                                        Storaget_1[storage_order_i] / storage_d_efficiencies[storage_order_i] / self.resolution)
                    Charget_update[storage_order_i] = np.minimum(np.minimum(Charget_node, S_p_capacity[storage_order_i] + SPower[t,storage_order_i]),
                                                        (S_e_capacity[storage_order_i] - Storaget_1[storage_order_i]) / storage_c_efficiencies[storage_order_i] / self.resolution)
                    
                    Discharget_i = SPower[t,storage_order_i] if SPower[t,storage_order_i] > 0 else 0
                    Charget_i = SPower[t,storage_order_i] if SPower[t,storage_order_i] < 0 else 0
                    Storaget[storage_order_i] = Storaget_1[storage_order_i] \
                                                - (Discharget_update[storage_order_i] + Discharget_i) * storage_d_efficiencies[storage_order_i] * self.resolution \
                                                + (Charget_update[storage_order_i] + Charget_i) * storage_c_efficiencies[storage_order_i] * self.resolution \

                    Discharget_node = Discharget_node - Discharget_update[storage_order_i]
                    Charget_node = Charget_node - Charget_update[storage_order_i]      

                # Flexible
                flexible_mask = self.flexible_nodes == node

                flexible_variable_costs_node = F_variable_costs[flexible_mask]    
                flexible_order_node = flexible_order[flexible_mask]
                sorted_flexible_indices = np.argsort(flexible_variable_costs_node)

                FPowert_node = FPowert[node]

                for i in range(len(sorted_flexible_indices)):
                    reverse_i = len(sorted_flexible_indices) - sorted_flexible_indices[i] - 1
                    flexible_order_i = flexible_order_node[reverse_i] - len(storage_order) - 1 # Adjust for storage orders

                    FPowert_update[flexible_order_i] = np.minimum(FPowert_node, Fcapacity[flexible_order_i] - FPower[t,flexible_order_i])
                    FPowert_node = FPowert_node - FPowert_update[flexible_order_i]

            SPower[t,:] = SPower[t,:] + Discharget_update - Charget_update
            FPower[t,:] = FPower[t,:] + FPowert_update
            Storage[t,:] = Storaget
            Storaget_1 = Storaget.copy()
            
            Transmission[t] = Transmissiont
            Discharge[t] = Discharget
            Charge[t] = Charget
        
        ImpExp = Transmission.sum(axis=1)
        
        Deficit = np.maximum(0, Netload - ImpExp - Discharge)
        Spillage = -1 * np.minimum(0, Netload - ImpExp + Charge)        

        self.Spillage_nodal = Spillage
        self.Storage = Storage
        self.Deficit_nodal = Deficit
        self.Import_nodal = np.maximum(0, ImpExp)
        self.Export_nodal = -1 * np.minimum(0, ImpExp)

        self.TFlows = (Transmission).sum(axis=2)

        #print(np.max(Spillage))
        np.savetxt("results/Deficit.csv", Deficit, delimiter=",")
        np.savetxt("results/Spillage.csv", Spillage, delimiter=",")
        np.savetxt("results/Storage.csv", Storage, delimiter=",")
        np.savetxt("results/Storage_Power.csv", SPower, delimiter=",")
        np.savetxt("results/Flexible.csv", FPower, delimiter=",")
        
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

        return None

    def _calculate_annual_generation(self):
        self.GPV_annual = self.GPV.sum(axis=0) * self.resolution / self.years
        self.GWind_annual = self.GWind.sum(axis=0) * self.resolution / self.years
        self.GFlexible_annual = self.GFlexible.sum(axis=0) * self.resolution / self.efficiency / self.years
        self.GBaseload_annual = self.GBaseload.sum(axis=0) * self.resolution / self.years
        self.TFlowsAbs_annual = self.TFlowsAbs.sum(axis=0) * self.resolution / self.years

        return None
    
    def _filter_balancing_profiles(self, Netload):
        balancing_p_profiles_ifft = np.zeros((self.intervals,self.nodes,max(self.nodal_balancing_count)), dtype=np.float64)
        
        """ np.savetxt("results/Charge_nodal.csv", self.Charge_nodal, delimiter=",")
        np.savetxt("results/Discharge_nodal.csv", self.Discharge_nodal, delimiter=",")
        np.savetxt("results/Storage_nodal.csv", self.Storage_nodal, delimiter=",") """
        """ np.savetxt("results/NetBalancing_nodal.csv", self.NetBalancing_nodal, delimiter=",") """
        
        for node_idx in range(self.nodes):
            frequency_profile_p = frequency.get_frequency_profile(Netload[:,node_idx])
            normalised_magnitudes_p = frequency.get_normalised_profile(frequency_profile_p)
            
            peak_mask, noise_mask = cwt.cwt_peak_detection(normalised_magnitudes_p)

            dc_offset_p = frequency.get_dc_offset(frequency_profile_p)
            dc_offset_p_timeseries = frequency.get_timeseries_profile(dc_offset_p)

            noise_frequency_profile = frequency.get_filtered_frequency(frequency_profile_p, noise_mask)
            noise_timeseries = frequency.get_timeseries_profile(noise_frequency_profile)
            
            for balancing_i in range(self.nodal_balancing_count[node_idx]):
                if abs(self.balancing_W_cutoffs[node_idx, balancing_i] - self.max_frequency) <= EPSILON_FLOAT64:
                    break

                #magnitude_filter_p = frequency.get_magnitude_filter(self.magnitudes_thresholds[node_idx, thresh_i-1],self.magnitudes_thresholds[node_idx, thresh_i], normalised_magnitudes_p)
                frequencies = frequency.get_frequencies(self.intervals, self.resolution)
                bandpass_filter = frequency.get_bandpass_filter(self.balancing_W_cutoffs[node_idx, balancing_i],self.balancing_W_cutoffs[node_idx, balancing_i+1], frequencies)
                peak_filter = bandpass_filter * peak_mask
                #np.savetxt(f"results/peak_filter_{balancing_i}.csv", peak_filter, delimiter=",")

                filtered_frequency_profile_p = frequency.get_filtered_frequency(frequency_profile_p, peak_filter)
                unit_timeseries_p = frequency.get_timeseries_profile(filtered_frequency_profile_p)
                balancing_p_profiles_ifft[:,node_idx,balancing_i] = unit_timeseries_p
                #np.savetxt(f"results/timeseries_{balancing_i}.csv", unit_timeseries_p, delimiter=",")
                
            # Apportion dc offset and noise to long-duration
            balancing_p_profiles_ifft[:, node_idx, :] = frequency.apportion_nodal_noise(balancing_p_profiles_ifft[:, node_idx, :], dc_offset_p_timeseries + noise_timeseries)
            
            """ np.savetxt(f"results/node_{node_idx}_timeseries.csv", self.balancing_p_profiles_ifft[:,node_idx,:], delimiter=",") """

        return balancing_p_profiles_ifft
    
    def _determine_constrained_balancing(self, balancing_p_profiles_ifft):
        #nodal_costs = NP_FLOAT_MAX * np.ones(self.nodes, dtype=np.float64)
        deficit = np.zeros(self.Deficit_nodal.shape, dtype=np.float64)
        surplus = np.zeros(self.Deficit_nodal.shape, dtype=np.float64)

        for node_idx in range(self.nodes): 
            #print(node_idx)
            node_mask = self.balancing_nodes == node_idx
            node_balancing_order = self.balancing_order[node_mask]

            node_balancing_e_capacities = self.balancing_e_constraints[node_balancing_order]
            variable_costs_per_mwh = self.balancing_costs[3,node_balancing_order] ##### ADD FUEL COSTS

            balancing_p_profile_ifft = balancing_p_profiles_ifft[:,node_idx,0:len(node_balancing_order)].copy()            

            node_balancing_permutation = frequency.order_balancing(node_balancing_order, 
                                                                   node_balancing_e_capacities,
                                                                   variable_costs_per_mwh)

            """ permutation_balancing_e_constraint = self.balancing_e_constraints[node_balancing_permutation]
            permutation_balancing_d_efficiencies = self.balancing_d_efficiencies[node_balancing_permutation]
            permutation_balancing_c_efficiencies = self.balancing_c_efficiencies[node_balancing_permutation] """
            permutation_balancing_d_constraints = self.balancing_d_constraint[node_balancing_permutation]
            permutation_balancing_c_constraints = self.balancing_c_constraint[node_balancing_permutation]
            """ permutation_balancing_costs = self.balancing_costs[:,node_balancing_permutation] """

            """ balancing_p_profile_option, balancing_e_profile_option, deficit_intranode, surplus_intranode = frequency.apply_balancing_constraints(balancing_p_profile_ifft, 
                                                                                                                permutation_balancing_e_constraint,
                                                                                                                permutation_balancing_d_efficiencies, 
                                                                                                                permutation_balancing_c_efficiencies, 
                                                                                                                permutation_balancing_d_constraints,
                                                                                                                permutation_balancing_c_constraints,
                                                                                                                self.resolution) """
            balancing_p_profile_option, deficit_intranode, surplus_intranode = frequency.apply_balancing_constraints(balancing_p_profile_ifft, 
                                                                                                                permutation_balancing_d_constraints,
                                                                                                                permutation_balancing_c_constraints)
                
            """ permutation_annual_discharge = helpers.sum_positive_values(balancing_p_profile_option) * self.resolution / self.years
            permutation_balancing_e_capacities = helpers.max_along_axis_n(balancing_e_profile_option, axis_n=0) / 1000 # MW to GW
            permutation_balancing_p_capacities = helpers.max_along_axis_n(np.abs(balancing_p_profile_option), axis_n=0) / 1000
                
            perm_cost = np.array([
                                annualisation_component(
                                    power_capacity=permutation_balancing_p_capacities[idx],
                                    energy_capacity=permutation_balancing_e_capacities[idx],
                                    annual_generation=permutation_annual_discharge[idx],
                                    capex_p=permutation_balancing_costs[0,idx],
                                    capex_e=permutation_balancing_costs[1,idx],
                                    fom=permutation_balancing_costs[2,idx],
                                    vom=permutation_balancing_costs[3,idx],
                                    lifetime=permutation_balancing_costs[4,idx],
                                    discount_rate=permutation_balancing_costs[5,idx]
                                ) for idx in range(0,len(permutation_balancing_e_capacities))
                                ], dtype=np.float64).sum()

            nodal_costs[node_idx] = perm_cost"""
            deficit[:,node_idx] = deficit_intranode
            surplus[:,node_idx] = surplus_intranode 

            for i in range(len(node_balancing_order)):
                flexible_order_offset = len(self.storage_ids) + 1
                if node_balancing_permutation[i] < flexible_order_offset - 1:
                    #self.storage_e_profiles[:,node_balancing_permutation[i]] = balancing_e_profile_option[:,i] 
                    self.storage_p_profiles[:,node_balancing_permutation[i]] = balancing_p_profile_option[:,i]
                    self.storage_p_capacities = self.CPHP
                    self.storage_e_capacities = self.CPHS
                else:
                    self.flexible_p_profiles[:,node_balancing_permutation[i]-flexible_order_offset] = balancing_p_profile_option[:,i]
                    self.flexible_p_capacities = self.CFlexible

                    """ print(f"Cheapest {node_idx}: {balancing_permutations[p,:]}") """

        """ print(self.storage_p_capacities, self.CPHS, self.flexible_p_capacities)
        np.savetxt("results/balancing_p_profiles_ifft.csv", self.balancing_p_profiles_ifft[:,0,:], delimiter=",")
        np.savetxt("results/GPV_nodal.csv", self.GPV_nodal, delimiter=",")
        np.savetxt("results/GWind_nodal.csv", self.GWind_nodal, delimiter=",")
        np.savetxt("results/GBaseload_nodal.csv", self.GBaseload_nodal, delimiter=",")
        np.savetxt("results/NetBalancing_nodal.csv", self.NetBalancing_nodal, delimiter=",")
        np.savetxt("results/storage_e_profiles.csv", self.storage_e_profiles, delimiter=",")
        np.savetxt("results/storage_p_profiles.csv", self.storage_p_profiles, delimiter=",")
        np.savetxt("results/flexible_p_profiles.csv", self.flexible_p_profiles, delimiter=",")
        np.savetxt("results/deficit_intranode.csv", deficit_intranodes, delimiter=",") """
        #exit()

        """ storage_costs = sum(nodal_costs)
        self.Deficit_nodal += deficit_intranodes
        self.Spillage_nodal += spillage_intranodes """
        
        return self.storage_p_profiles, self.flexible_p_profiles, deficit, surplus

    def _objective(self) -> List[float]:
        deficit_nodal = self._reliability() 
        pen_deficit = np.maximum(0., deficit_nodal.sum() * self.resolution - self.allowance) * 1000
        self.TFlowsAbs = np.abs(self.TFlows)

        """ self._filter_balancing_profiles()
        storage_costs, deficit_intranode = self._determine_cheapest_balancing()
        pen_deficit = np.maximum(0., (deficit_nodal + deficit_intranode).sum() * self.resolution - self.allowance) * 1000 """

        self._apportion_nodal_generation()
        self._calculate_annual_generation()
        cost = self._calculate_costs() ####### ADD STORAGE AND FLEXIBLE COSTS BACK IN
        #cost += storage_costs

        loss = self.TFlowsAbs.sum(axis=0) * self.TLoss
        loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - loss)
        
        print("LCOE: ", lcoe, pen_deficit)
        exit()
        return lcoe, pen_deficit

    def evaluate(self):
        self.lcoe, self.penalties = self._objective()
        self.evaluated=True 
        return self

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
                    storage_unit_types,
                    flexible_ids,
                    nodes_with_balancing,
                    max_frequency,
                    storage_durations,
                    storage_costs,
                    line_ids,
                    line_lengths,
                    line_costs,
                    TLoss,
                    pv_idx,
                    wind_idx,
                    flexible_p_idx,
                    storage_p_idx,
                    storage_e_idx,
                    lines_idx,
                    balancing_W_idx,
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
                    storage_cost_ids,
                    line_cost_ids,
                    networksteps,
                    storage_d_efficiencies,
                    storage_c_efficiencies,):
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
                                storage_unit_types,
                                flexible_ids,
                                nodes_with_balancing,
                                max_frequency,
                                storage_durations,
                                storage_costs,
                                line_ids,
                                line_lengths,
                                line_costs,
                                TLoss,
                                pv_idx,
                                wind_idx,
                                flexible_p_idx,
                                storage_p_idx,
                                storage_e_idx,
                                lines_idx,
                                balancing_W_idx,
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
                                storage_cost_ids,
                                line_cost_ids,
                                networksteps,
                                storage_d_efficiencies,
                                storage_c_efficiencies,)
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
                storage_unit_types,
                flexible_ids,
                nodes_with_balancing,
                max_frequency,
                storage_durations,
                storage_costs,
                line_ids,
                line_lengths,
                line_costs,
                TLoss,
                pv_idx,
                wind_idx,
                flexible_p_idx,
                storage_p_idx,
                storage_e_idx,
                lines_idx,
                balancing_W_idx,
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
                storage_cost_ids,
                line_cost_ids,
                networksteps,
                storage_d_efficiencies,
                storage_c_efficiencies,):
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
                                storage_unit_types,
                                flexible_ids,
                                nodes_with_balancing,
                                max_frequency,
                                storage_durations,
                                storage_costs,
                                line_ids,
                                line_lengths,
                                line_costs,
                                TLoss,
                                pv_idx,
                                wind_idx,
                                flexible_p_idx,
                                storage_p_idx,
                                storage_e_idx,
                                lines_idx,
                                balancing_W_idx,
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
                                storage_cost_ids,
                                line_cost_ids,
                                networksteps,
                                storage_d_efficiencies,
                                storage_c_efficiencies,)
    solution.evaluate()
    return solution.lcoe + solution.penalties