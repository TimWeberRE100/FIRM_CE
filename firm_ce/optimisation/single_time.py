import numpy as np
from typing import List
import time

from firm_ce.system.topology import get_transmission_flows_t
from firm_ce.common.constants import JIT_ENABLED, EPSILON_FLOAT64
from firm_ce.system.costs import calculate_costs
import firm_ce.common.helpers as helpers

if JIT_ENABLED:
    from numba import float64, int64, boolean, njit, prange
    from numba.experimental import jitclass

    solution_spec = [
        ('x', float64[:]),
        ('evaluated', boolean),
        ('lcoe', float64),
        ('penalties', float64),

        ('MLoad', float64[:, :]),
        ('loss', float64),
        ('intervals', int64),
        ('nodes', int64),
        ('lines', int64),
        ('resolution', float64),
        ('years', int64),
        ('energy', float64),
        ('allowance', float64),
        ('year_first_t', int64[:]),

        # Generators
        ('generator_ids', int64[:]),
        ('generator_costs', float64[:, :]),
        ('CBaseload', float64[:]),
        ('GBaseload', float64[:, :]),
        ('GFlexible', float64[:, :]),
        ('Flexible_Limits_Annual', float64[:, :]),
        ('GFlexible_constraint', float64[:, :]),
        ('_Flexible_max', float64[:]),

        # Storages
        ('storage_ids', int64[:]),
        ('storage_nodes', int64[:]),
        ('GDischarge', float64[:, :]),
        ('Storage', float64[:, :]),
        ('SPower', float64[:, :]),
        ('_Charget_max', float64[:]),
        ('_Discharget_max', float64[:]),

        # Balancing
        ('flexible_ids', int64[:]),
        ('storage_order', int64[:]),
        ('flexible_order', int64[:]),
        ('storage_nodal_count', int64[:]),
        ('flexible_nodal_count', int64[:]),
        ('storage_durations', float64[:]),
        ('storage_costs', float64[:, :]),        
        ('storage_d_efficiencies', float64[:]),
        ('storage_c_efficiencies', float64[:]),

        # Lines
        ('line_ids', int64[:]),
        ('line_lengths', float64[:]),
        ('line_costs', float64[:, :]),
        ('network', int64[:, :, :, :]),
        ('networksteps', int64),

        ('TLoss', float64[:]),
        ('TFlows', float64[:, :]),
        ('TFlowsAbs', float64[:, :]),

        # Decision Variables
        ('CPV', float64[:]),
        ('CWind', float64[:]),
        ('CFlexible', float64[:]),
        ('CPHP', float64[:]),
        ('CPHS', float64[:]),
        ('CTrans', float64[:]),

        # Transmission
        ('Transmission', float64[:, :, :]),
        ('trans_tflows_mask', boolean[:,:,:]),

        # Nodal assignments
        ('flexible_nodes', int64[:]),
        ('baseload_nodes', int64[:]),

        ('CFlexible_nodal', float64[:]),
        ('GFlexible_nodal', float64[:, :]),
        ('CPHP_nodal', float64[:]),
        ('CPHS_nodal', float64[:]),
        ('GPV', float64[:, :]),
        ('GPV_nodal', float64[:, :]),
        ('GWind', float64[:, :]),
        ('GWind_nodal', float64[:, :]),
        ('CBaseload_nodal', float64[:]),
        ('GBaseload_nodal', float64[:, :]),
        ('Spillage_nodal', float64[:, :]),
        ('NetBalancing_nodal', float64[:, :]),
        ('Deficit_nodal', float64[:, :]),
        ('SPower_nodal', float64[:, :]),
        ('_Charget_max_nodal', float64[:]),
        ('_Discharget_max_nodal', float64[:]),
        ('_Flexible_max_nodal', float64[:]),
        ('storage_sorted_nodal', int64[:, :]),
        ('flexible_sorted_nodal', int64[:, :]),
        ('flexible_sorted', int64[:]),

        ('GPV_annual', float64[:]),
        ('GWind_annual', float64[:]),
        ('GFlexible_annual', float64[:]),
        ('Flexible_hours_annual', float64[:]),
        ('GBaseload_annual', float64[:]),
        ('GDischarge_annual', float64[:]),
        ('TFlowsAbs_annual', float64[:]),

        ('pv_cost_ids', int64[:]),
        ('wind_cost_ids', int64[:]),
        ('flexible_cost_ids', int64[:]),
        ('baseload_cost_ids', int64[:]),
        ('storage_cost_ids', int64[:]),
        ('line_cost_ids', int64[:]),

        ('generator_line_ids', int64[:]),
        ('storage_line_ids', int64[:])
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
                TSBaseload,
                network,
                transmission_mask,
                intervals,
                nodes,
                lines,
                years,
                resolution,
                allowance,
                generator_ids,
                generator_costs,
                storage_ids,
                storage_nodes,
                flexible_ids,
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
                solar_nodes,
                wind_nodes,
                flexible_nodes,
                baseload_nodes,
                CBaseload,
                pv_cost_ids,
                wind_cost_ids,
                flexible_cost_ids,
                baseload_cost_ids,
                storage_cost_ids,
                line_cost_ids,
                networksteps,
                storage_d_efficiencies,
                storage_c_efficiencies,
                Flexible_Limits_Annual,
                first_year,
                generator_line_ids,
                storage_line_ids) -> None:

        self.x = x  
        self.evaluated=False   
        self.lcoe = 0.0
        self.penalties = 0.0

        self.MLoad = MLoad / 1000 # MW to GW
        self.loss = 0.0
        self.intervals = intervals
        self.nodes = nodes
        self.lines = lines
        self.resolution = resolution
        self.years = years
        self.year_first_t = self._get_year_t_arr(first_year)
        self.energy = self.MLoad.sum() * self.resolution / self.years
        self.allowance = allowance*self.energy

        # Generators
        self.generator_ids = generator_ids
        self.generator_line_ids = generator_line_ids
        self.generator_costs = generator_costs

        self.CBaseload = CBaseload + EPSILON_FLOAT64 # Prevent 0GW capacities from being excluded from statistics
        self.GBaseload = self.CBaseload * TSBaseload
        self.GFlexible = np.zeros((self.intervals, len(flexible_ids)), dtype=np.float64)
        self.Flexible_Limits_Annual = Flexible_Limits_Annual
        self.GFlexible_constraint = np.zeros((self.intervals,len(flexible_ids)), dtype=np.float64)
        self._Flexible_max = np.zeros(len(flexible_ids), dtype=np.float64)

        # Storages
        self.storage_ids = storage_ids
        self.storage_line_ids = storage_line_ids
        self.storage_nodes = storage_nodes
        self.storage_durations = storage_durations
        self.storage_costs = storage_costs
        self.storage_d_efficiencies = storage_d_efficiencies
        self.storage_c_efficiencies = storage_c_efficiencies

        self.Storage = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)
        self.SPower = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)
        self.GDischarge = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)
        self._Charget_max = np.zeros(len(storage_ids), dtype=np.float64)
        self._Discharget_max = np.zeros(len(storage_ids), dtype=np.float64)

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
        x += EPSILON_FLOAT64 # Prevent 0GW capacities from being excluded from statistics
        self.CPV = x[: pv_idx]
        self.CWind = x[pv_idx : wind_idx]
        self.CFlexible = x[wind_idx : flexible_p_idx]
        self.CPHP = x[flexible_p_idx : storage_p_idx]
        self.CPHS = x[storage_p_idx : storage_e_idx]
        self.CTrans = x[storage_e_idx : lines_idx]

        for idx in range(len(storage_durations)):
            if storage_durations[idx] > 0:
                self.CPHS[idx] = self.CPHP[idx] * storage_durations[idx]

        """ print(pv_idx,wind_idx,flexible_p_idx,storage_p_idx,storage_e_idx,lines_idx)
        print(self.CPV)
        print(self.CWind)
        print(self.CFlexible)
        print(self.CPHP)
        print(self.CPHS)
        print(self.CTrans)
        print(self.GBaseload)
        print(self.years)
        print(self.Flexible_Limits_Annual)
        print(self.GFlexible_constraint)
        print(self._Flexible_max) """

        # Transmission
        self.Transmission = np.zeros((self.intervals, len(self.CTrans), self.nodes), dtype = np.float64)
        self.trans_tflows_mask = transmission_mask

        # Nodal Values
        self.flexible_nodes = flexible_nodes 
        self.baseload_nodes = baseload_nodes

        self.CFlexible_nodal = self._fill_nodal_array_1d(self.CFlexible, self.flexible_nodes)
        self.CPHP_nodal = self._fill_nodal_array_1d(self.CPHP, self.storage_nodes)
        self.CPHS_nodal = self._fill_nodal_array_1d(self.CPHS, self.storage_nodes)    
        self.GPV = self.CPV[np.newaxis, :] * TSPV
        self.GPV_nodal = self._fill_nodal_array_2d(self.GPV, solar_nodes)
        self.GWind = self.CWind[np.newaxis, :] * TSWind
        self.GWind_nodal = self._fill_nodal_array_2d(self.GWind, wind_nodes)
        self.CBaseload_nodal = self._fill_nodal_array_1d(self.CBaseload, self.baseload_nodes)
        self.GBaseload_nodal = self._fill_nodal_array_2d(self.GBaseload, self.baseload_nodes)
        
        self.SPower_nodal=  np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.GFlexible_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Spillage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Deficit_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64) 

        self._Charget_max_nodal = np.zeros(self.nodes, dtype=np.float64) 
        self._Discharget_max_nodal = np.zeros(self.nodes, dtype=np.float64) 
        self._Flexible_max_nodal = np.zeros(self.nodes, dtype=np.float64) 

        self.GPV_annual = np.zeros(self.CPV.shape, dtype=np.float64)
        self.GWind_annual = np.zeros(self.CWind.shape, dtype=np.float64)
        self.GFlexible_annual = np.zeros(self.CFlexible.shape, dtype=np.float64)
        self.Flexible_hours_annual = np.zeros(self.CFlexible.shape, dtype=np.float64)
        self.GBaseload_annual = np.zeros(self.CBaseload.shape, dtype=np.float64)
        self.GDischarge_annual = np.zeros(self.CPHP.shape, dtype=np.float64)
        self.TFlowsAbs_annual = np.zeros(self.CTrans.shape, dtype=np.float64)

        # Balancing
        self.storage_order = np.arange(len(storage_ids), dtype=np.int64)
        self.flexible_order = np.arange(len(flexible_ids), dtype=np.int64)
        self.storage_nodal_count = np.zeros(nodes, dtype=np.int64)
        self.flexible_nodal_count = np.zeros(nodes, dtype=np.int64)
        for node in storage_nodes:
            self.storage_nodal_count[node] += 1
        for node in flexible_nodes:
            self.flexible_nodal_count[node] += 1

        self.storage_sorted_nodal = -1*np.ones((nodes,max(self.storage_nodal_count)), dtype=np.int64)
        self.flexible_sorted_nodal = -1*np.ones((nodes,max(self.flexible_nodal_count)), dtype=np.int64)

        flexible_mask = helpers.isin_numba(np.arange(self.generator_costs.shape[1], dtype=np.int64), flexible_cost_ids)
        F_variable_costs = (self.generator_costs[3, flexible_mask] + self.generator_costs[6, flexible_mask]) + self.generator_costs[7, flexible_mask] # SRMC of 1 MWh in 1 h
        self.flexible_sorted = np.argsort(F_variable_costs)

        for node in range(nodes):
            storage_mask = self.storage_nodes == node
            if np.any(storage_mask):
                sorted_indices = np.argsort((self.CPHS/self.CPHP)[storage_mask])

                for i in range(sorted_indices.shape[0]):
                    self.storage_sorted_nodal[node, i] = sorted_indices[i]
                        
            flexible_mask = self.flexible_nodes == node
            if np.any(flexible_mask):
                sorted_indices = np.argsort(F_variable_costs[flexible_mask])

                for i in range(sorted_indices.shape[0]):
                    self.flexible_sorted_nodal[node, i] = sorted_indices[i]

        # Cost Values
        self.pv_cost_ids = pv_cost_ids
        self.wind_cost_ids = wind_cost_ids
        self.flexible_cost_ids = flexible_cost_ids
        self.baseload_cost_ids = baseload_cost_ids
        self.storage_cost_ids = storage_cost_ids
        self.line_cost_ids = line_cost_ids

    def _get_year_t_arr(self,first_year):
        year_first_t = np.zeros(self.years, dtype=np.int64)

        for i in range(self.years):
            year = first_year + i

            first_t = i * (8760 // self.resolution)

            leap_days = 0
            for y in range(first_year, year + 1):
                if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
                    leap_days += 1

            leap_adjust = leap_days * (24 // self.resolution)

            year_first_t[i] = first_t + leap_adjust

        return year_first_t

    def _fill_nodal_array_2d(self, generation_array, node_array):
        result = np.zeros((self.intervals, self.nodes), dtype=np.float64)

        for t in range(self.intervals):
            for n, node_idx in enumerate(node_array):
                result[t, node_idx] += generation_array[t, n]

        return result
    
    def _fill_nodal_array_1d(self, capacity_array, node_array):
        result = np.zeros(self.nodes, dtype=np.float64)

        for n, node_idx in enumerate(node_array):
            result[node_idx] += capacity_array[n]

        return result
    
    def _apportion_nodal_array(self, capacity_array, nodal_generation, node_array):        
        cap_node_sum = np.zeros(self.nodes, dtype=np.float64)
        for i in range(len(node_array)):
            cap_node_sum[node_array[i]] += capacity_array[i]
        
        result = np.zeros((self.intervals, len(node_array)), dtype=np.float64)
        
        for i in range(len(node_array)):
            if cap_node_sum[node_array[i]] > 0.0:
                result[:,i] = capacity_array[i] / cap_node_sum[node_array[i]] * nodal_generation[:, node_array[i]]
            else:
                result[:,i] = nodal_generation[:, node_array[i]]

        return result
    
    def _get_flexible_hours(self):
        flexible_hours = np.zeros(len(self.flexible_order), dtype=np.float64)
        for i in range(self.intervals):
            for f in self.flexible_order:
                if self.GFlexible[i,f] > 1e-6:
                    flexible_hours[f] += self.resolution

        return flexible_hours

    def _calculate_annual_generation(self):
        self.GPV_annual = self.GPV.sum(axis=0) * self.resolution / self.years
        self.GWind_annual = self.GWind.sum(axis=0) * self.resolution / self.years
        self.GDischarge_annual = self.GDischarge.sum(axis=0) * self.resolution / self.years
        self.GFlexible_annual = self.GFlexible.sum(axis=0) * self.resolution / self.years
        self.Flexible_hours_annual = self._get_flexible_hours() * self.resolution / self.years
        self.GBaseload_annual = self.GBaseload.sum(axis=0) * self.resolution / self.years
        self.TFlowsAbs_annual = self.TFlowsAbs.sum(axis=0) * self.resolution / self.years

        return None
    
    def _update_storage(self, t, Storaget_1, forwards_t=True):
        if forwards_t:
            return (
                Storaget_1
                - np.maximum(self.SPower[t], 0) / self.storage_d_efficiencies * self.resolution
                - np.minimum(self.SPower[t], 0) * self.storage_c_efficiencies * self.resolution
            )
        # Reverse charge/discharge effect in reverse time
        else:
            return (
                Storaget_1
                + np.maximum(self.SPower[t], 0) / self.storage_d_efficiencies * self.resolution
                + np.minimum(self.SPower[t], 0) * self.storage_c_efficiencies * self.resolution
            )
    
    def _clamp_and_assign(self, t, node, idx, is_flexible=False, lower_bound=None, upper_bound=None):
        if is_flexible:
            # For flexible generation, lower bound is 0.
            lower = 0 if lower_bound is None else lower_bound
            upper = self._Flexible_max[idx] if upper_bound is None else upper_bound
            current = self.GFlexible_nodal[t][node]
            new_value = helpers.scalar_clamp(current, lower, upper)
            self.GFlexible_nodal[t][node] -= new_value
            self.GFlexible[t][idx] = new_value
        else:
            # For storage dispatch, use passed bounds (for charging/discharging).
            lower = -self._Charget_max[idx] if lower_bound is None else lower_bound
            upper = self._Discharget_max[idx] if upper_bound is None else upper_bound
            current = self.SPower_nodal[t][node]
            new_value = helpers.scalar_clamp(current, lower, upper)
            self.SPower_nodal[t][node] -= new_value
            self.SPower[t][idx] = new_value

        return None
    
    def _determine_precharge_energies(self, t, Netload):
        Storaget_1_reversed = self.Storage[t-1].copy() 
        Flexible_Limit_reversed = self.GFlexible_constraint[t-1].copy()
        Flexible_Limit_reversed2 = Flexible_Limit_reversed.copy()
        local_min = Storaget_1_reversed.copy() 
        local_max = Storaget_1_reversed.copy()      
        
        while True:
            t -= 1

            self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)
            self.GFlexible_nodal[t] = self._fill_nodal_array_1d(self.GFlexible[t], self.flexible_nodes)            
            
            Storaget_p_lb_rev = Storaget_1_reversed / self.storage_c_efficiencies / self.resolution 
            Storaget_p_ub_rev = (self.CPHS - Storaget_1_reversed) * self.storage_d_efficiencies / self.resolution
            self._Discharget_max = np.minimum(self.CPHP, Storaget_p_ub_rev) # Reversed energy constraint in reverse time
            self._Discharget_max_nodal = self._fill_nodal_array_1d(self._Discharget_max, self.storage_nodes)
            self._Charget_max = np.minimum(self.CPHP, Storaget_p_lb_rev) # Reversed energy constraint in reverse time
            self._Charget_max_nodal = self._fill_nodal_array_1d(self._Charget_max, self.storage_nodes)

            if t-1 in self.year_first_t:
                for i in range(len(self.year_first_t)):
                    if t-1 == self.year_first_t[i]:
                        Flexiblet_p_lb = (self.Flexible_Limits_Annual[i]) / self.resolution
            else:
                for i in range(len(self.year_first_t)):
                    if t-1 > self.year_first_t[i]:
                        Flexiblet_p_lb = (self.Flexible_Limits_Annual[i] - Flexible_Limit_reversed) / self.resolution

            self._Flexible_max = np.minimum(self.CFlexible, Flexiblet_p_lb)# Reversed energy constraint in reverse time
            self._Flexible_max_nodal = self._fill_nodal_array_1d(self._Flexible_max, self.flexible_nodes)

            self.Transmission[t] = np.zeros((len(self.CTrans), self.nodes), dtype = np.float64) # Reset the transmission

            Netloadt = Netload[t].copy()  

            # Avoiding extra memory allocation. This will be overwritten after loop
            self.Deficit_nodal[t] = np.maximum(Netloadt, 0)

            if self.Deficit_nodal[t].sum() > 1e-6:
                # Fill deficits with transmitted spillage               
                self.Spillage_nodal[t] = (
                    -1 * np.minimum(0, Netloadt)
                )

                self.Transmission[t] = get_transmission_flows_t(
                    self.Deficit_nodal[t], self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt -= self.Transmission[t].sum(axis=0)

            self.Deficit_nodal[t] = np.maximum(Netloadt, 0)
            if self.Deficit_nodal[t].sum() > 1e-6:
                # Draw down from neighbours storage and flexible reserves                
                self.Spillage_nodal[t] = (
                    self._Discharget_max_nodal
                    + self.CFlexible_nodal
                )

                self.Transmission[t] = get_transmission_flows_t(
                    self.Deficit_nodal[t], self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                self.SPower_nodal[t] = (
                    np.maximum(np.minimum(Netloadt, self._Discharget_max_nodal),0) +
                    np.minimum(np.maximum(Netloadt, -self._Charget_max_nodal),0)                     
                ) 

                self.GFlexible_nodal[t] = np.minimum(
                    np.maximum(Netloadt - self.SPower_nodal[t], 0),
                    self._Flexible_max_nodal
                ) 

            self.Spillage_nodal[t] = -1 * np.minimum(0, Netloadt - np.minimum(self.SPower_nodal[t], 0))
            if self.Spillage_nodal[t].sum() > 1e-6:
                self.Transmission[t] = get_transmission_flows_t(
                    (self.SPower_nodal[t] + self._Charget_max_nodal), 
                    self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                self.SPower_nodal[t] = (
                    np.maximum(np.minimum(Netloadt, self._Discharget_max_nodal),0) +
                    np.minimum(np.maximum(Netloadt, -self._Charget_max_nodal),0) 
                ) 

                self.GFlexible_nodal[t] = np.minimum(
                    np.maximum(Netloadt - self.SPower_nodal[t], 0),
                    self._Flexible_max_nodal
                ) 

            # Apportion to individual storages/flexible 
            self.Deficit_nodal[t] = np.maximum(Netloadt - self.SPower_nodal[t] - self.GFlexible_nodal[t], 0) 
                
            for node in range(self.nodes):
                # Apportion storage
                storage_mask = self.storage_nodes == node
                if np.any(storage_mask):
                    for idx in self.storage_sorted_nodal[node,::-1]:
                        if idx == -1:
                            break
                        self._clamp_and_assign(t, node, self.storage_order[storage_mask][idx])

                        """ if idx == 0:
                            break """

                # Apportion flexible
                flexible_mask = self.flexible_nodes == node
                if np.any(flexible_mask):
                    for idx in self.flexible_sorted_nodal[node,:]: 
                        if idx == -1:
                            break      
                        self._clamp_and_assign(t, node, self.flexible_order[flexible_mask][idx], True)

                        """ if idx == 0:
                            break """
            
            Storaget_1_reversed = self._update_storage(t, Storaget_1_reversed, False)
            Flexible_Limit_reversed += self.GFlexible[t] * self.resolution
            
            local_max = np.maximum(local_max, Storaget_1_reversed)
            local_min = np.minimum(local_min, Storaget_1_reversed)

            # If you reach end of deficit block, return results
            if ((Netload[t-1] - self.SPower_nodal[t-1] - self.GFlexible_nodal[t-1] - self.Transmission[t-1].sum(axis=0) < 1e-6).all() or t < 1):
                Storaget_1_forward = self._update_storage(t, self.Storage[t-1])
                
                local_max = np.maximum(local_max, Storaget_1_forward)
                local_min = np.minimum(local_min, Storaget_1_forward)

                precharge_energy = (Storaget_1_reversed - Storaget_1_forward)
                precharge_mask = (local_max - local_min > Storaget_1_forward)
                precharge_energy[~precharge_mask] = 0
                trickling_reserves = local_max - local_min
                flexible_reserves = Flexible_Limit_reversed - Flexible_Limit_reversed2

                return trickling_reserves, precharge_energy, flexible_reserves, t+1
    
    def _get_energy_change(self, original_power, change_power, storage_idx):
        # Energy change sense based on forward time relationship (t increasing)
        change_energy = 0.
        # Original discharging
        if original_power > 0:
            # Increase discharging power
            if change_power > 0:
                change_energy = -change_power / self.storage_d_efficiencies[storage_idx] / self.resolution
            # Reduce discharging power and increase charging power
            else:
                change_energy = (min(original_power, -change_power) / self.storage_d_efficiencies[storage_idx] / self.resolution - 
                                 min(original_power + change_power, 0.) * self.storage_c_efficiencies[storage_idx] / self.resolution)
        # Original charging
        elif original_power < 0:
            # Reduce charging power and increase discharging power
            if change_power > 0:
                change_energy = (-max(original_power, -change_power) * self.storage_c_efficiencies[storage_idx] / self.resolution + 
                                 max(original_power + change_power, 0.) / self.storage_d_efficiencies[storage_idx] / self.resolution)
            # Increase charging power
            else:
                change_energy = -change_power * self.storage_c_efficiencies[storage_idx] / self.resolution
        
        else:
            if change_power > 0:
                change_energy = -change_power / self.storage_d_efficiencies[storage_idx] / self.resolution
            else:
                change_energy = -change_power * self.storage_c_efficiencies[storage_idx] / self.resolution

        return change_energy
    
    def _get_change_power_bounds(self, trickling_reserves, precharge_energy, t, flexible_bool):
        if not flexible_bool:
            # Trickling
            charge_reduction_constraint = np.minimum(trickling_reserves / self.storage_c_efficiencies / self.resolution, 
                                                    -np.minimum(self.SPower[t], 0))
            
            discharge_increase_constraint = np.minimum((trickling_reserves - charge_reduction_constraint*self.storage_c_efficiencies*self.resolution) * self.storage_d_efficiencies / self.resolution, 
                                                    self.CPHP - np.maximum(self.SPower[t], 0))
            
            trickling_change_max = charge_reduction_constraint + discharge_increase_constraint

        else:
            # Trickling
            trickling_change_max = np.minimum(self.CFlexible - self.GFlexible[t], trickling_reserves / self.resolution) 
        
        # Precharging
        discharge_reduction_constraint = np.minimum(precharge_energy * self.storage_d_efficiencies / self.resolution,
                                                    np.maximum(self.SPower[t], 0))
            
        charge_increase_constraint = np.minimum((precharge_energy - discharge_reduction_constraint/self.storage_d_efficiencies*self.resolution) / self.storage_c_efficiencies / self.resolution,
                                                self.CPHP + np.minimum(self.SPower[t], 0))
            
        precharging_change_max = discharge_reduction_constraint + charge_increase_constraint

        return trickling_change_max, precharging_change_max
    
    def _determine_precharge_powers(self, trickling_reserves, precharge_energy, flexible_reserves, t, Netload):
        t_precharge_start = 0
        trickling_mask = (self.Storage[t] - trickling_reserves > 1e-6) & (precharge_energy < 1e-6)  
        precharging_mask = (precharge_energy > 1e-6) 
        flexible_trickling_mask = (self.GFlexible_constraint[t] - flexible_reserves > 1e-6)

        while True:
            t -= 1    

            remaining_trickling_reserves = np.maximum(self.Storage[t] - trickling_reserves,0)             
            trickling_mask = (remaining_trickling_reserves > 1e-6) & trickling_mask             
            precharging_mask = (self.Storage[t] + 1e-6 < self.CPHS) & (precharge_energy > 1e-6) & precharging_mask 

            trickling_change_max, precharging_change_max = self._get_change_power_bounds(remaining_trickling_reserves, precharge_energy, t, False)
            if not trickling_mask.all():
                trickling_change_max[~trickling_mask] = 0
            if not precharging_mask.all():
                precharging_change_max[~precharging_mask] = 0
            trickling_max_nodal = self._fill_nodal_array_1d(trickling_change_max, self.storage_nodes)
            precharging_max_nodal = self._fill_nodal_array_1d(precharging_change_max, self.storage_nodes)            

            Transmissiont_pre = self.Transmission[t].sum(axis=0)
            self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)
            self.GFlexible_nodal[t] = self._fill_nodal_array_1d(self.GFlexible[t], self.flexible_nodes)
            
            Surplust = trickling_max_nodal.copy()
            Fillt = precharging_max_nodal.copy()

            # Spillage intranode precharging
            Netloadt = Netload[t] - self.SPower_nodal[t] - self.GFlexible_nodal[t] - Transmissiont_pre
            self.Spillage_nodal[t] = -1 * np.minimum(0, Netloadt - np.minimum(self.SPower_nodal[t], 0))
            for i in range(len(self.storage_nodes)):
                if precharging_mask[i]:
                    node = self.storage_nodes[i]

                    change_power = min(max(-self.Spillage_nodal[t,node],-precharging_change_max[i]),0)

                    change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                    self.SPower[t,i] += change_power
                    self.Spillage_nodal[t,node] -= change_power
                    Fillt[node] -= change_power
                    precharging_change_max[i] -= change_power
                    precharge_energy[i] -= change_energy  

            # Spillage internode precharging
            if self.Spillage_nodal[t].sum() > 1e-6:
                self.Transmission[t] = get_transmission_flows_t(
                    Fillt, 
                    self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Transmissiont_pre - self.Transmission[t].sum(axis=0)
                Transmissiont_pre = self.Transmission[t].sum(axis=0)

                for i in range(len(self.storage_nodes)):
                    node = self.storage_nodes[i]

                    if precharging_mask[i]:
                        change_power = min(max(Netloadt[node], -precharging_change_max[i]),0)
                        change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                        self.SPower[t,i] += change_power
                        Netloadt[node] -= change_power
                        Fillt[node] -= change_power
                        precharging_change_max[i] -= change_power
                        precharge_energy[i] -= change_energy  

                self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)

            # Intranode precharging
            IntranodeCharget = np.minimum(Surplust, Fillt)
            Intranodet_trickle = IntranodeCharget.copy()
            Intranodet_charge = -IntranodeCharget.copy()

            Fillt -= IntranodeCharget
            Surplust -= IntranodeCharget

            for i in range(len(self.storage_nodes)):
                if not trickling_mask[i] and not precharging_mask[i]:
                    continue

                node = self.storage_nodes[i]

                if trickling_mask[i]:
                    change_power = max(min(Intranodet_trickle[node],trickling_change_max[i]),0)

                    change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                    self.SPower[t,i] += change_power
                    Intranodet_trickle[node] -= change_power 
                    trickling_reserves[i] -= change_energy   

                if precharging_mask[i]:
                    change_power = min(max(Intranodet_charge[node],-precharging_change_max[i]),0)

                    change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                    self.SPower[t,i] += change_power
                    Intranodet_charge[node] -= change_power
                    precharge_energy[i] -= change_energy  

            self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)
            
            remaining_trickling_reserves = np.maximum(self.Storage[t] - trickling_reserves,0) 
            trickling_mask = (remaining_trickling_reserves > 1e-6) & trickling_mask  
            precharging_mask = (self.Storage[t] + 1e-6 < self.CPHS) & (precharge_energy > 1e-6) & precharging_mask 
            
            # Internode precharging
            if (Fillt.sum() > 1e-6) and (Surplust.sum() > 1e-6):
                self.Transmission[t] = get_transmission_flows_t(
                    Fillt, Surplust, self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Transmissiont_pre - self.Transmission[t].sum(axis=0)

                for i in range(len(self.storage_nodes)):
                    node = self.storage_nodes[i]

                    if trickling_mask[i]:
                        change_power = max(min(Netloadt[node],trickling_change_max[i]),0)
                        change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                        self.SPower[t,i] += change_power
                        Netloadt[node] -= change_power
                        trickling_reserves[i] += change_energy

                    if precharging_mask[i]:
                        change_power = min(max(Netloadt[node], -precharging_change_max[i]),0)
                        change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                        self.SPower[t,i] += change_power
                        Netloadt[node] -= change_power
                        precharge_energy[i] -= change_energy  

                self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes) 

            if (precharge_energy > 1e-6).any():
                # Use flexible if all trickling reserves have been used
                precharge_energy, flexible_reserves, flexible_trickling_mask = self._determine_flexible_precharging(t, precharging_mask, precharge_energy, flexible_reserves)
                
            if (precharge_energy < 1e-6).all() or (not precharging_mask.any()) or (t < 1) or (not trickling_mask.any() and not flexible_trickling_mask.any()):
                t_precharge_start = t
                break
                
        return t_precharge_start
    
    def _determine_flexible_precharging(self, t, precharging_mask, precharge_energy, flexible_reserves):
        # Remaining_flexible_reserves
        remaining_flexible_reserves = np.maximum(self.GFlexible_constraint[t] - flexible_reserves,0) 
        flexible_trickling_mask = (remaining_flexible_reserves > 1e-6)
        precharging_mask = (self.Storage[t] + 1e-6 < self.CPHS) & (precharge_energy > 1e-6) & precharging_mask       

        trickling_change_max, precharging_change_max = self._get_change_power_bounds(remaining_flexible_reserves, precharge_energy, t, True)
        if not flexible_trickling_mask.all():
            trickling_change_max[~flexible_trickling_mask] = 0 
        if not precharging_mask.all():
            precharging_change_max[~precharging_mask] = 0 
        trickling_max_nodal = self._fill_nodal_array_1d(trickling_change_max, self.flexible_nodes)
        precharging_max_nodal = self._fill_nodal_array_1d(precharging_change_max, self.storage_nodes)            

        Transmissiont_pre = self.Transmission[t].sum(axis=0)
            
        Surplust = trickling_max_nodal.copy()
        Fillt = precharging_max_nodal.copy()

        # Intranode precharging
        IntranodeCharget = np.minimum(Surplust, Fillt)
        Intranodet_trickle = IntranodeCharget.copy()
        Intranodet_charge = -IntranodeCharget.copy()

        Fillt -= IntranodeCharget
        Surplust -= IntranodeCharget

        for i in range(len(self.flexible_nodes)): 
            fidx = self.flexible_sorted[i]
            
            if flexible_trickling_mask[fidx]:
                node = self.flexible_nodes[fidx]

                change_power = max(min(Intranodet_trickle[node],trickling_change_max[fidx]),0)
                change_energy = -change_power / self.resolution

                self.GFlexible[t,fidx] += change_power
                flexible_reserves[fidx] -= change_energy 
                Intranodet_trickle[node] -= change_power  
                trickling_change_max[fidx] -= change_power         

        for i in range(len(self.storage_nodes)):
            if precharging_mask[i]:
                node = self.storage_nodes[i] 

                change_power = min(max(Intranodet_charge[node],-precharging_change_max[i]),0)
                change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                self.SPower[t,i] += change_power
                Intranodet_charge[node] -= change_power
                precharge_energy[i] -= change_energy 
                precharging_change_max[i] -= change_power

        self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)
        self.GFlexible_nodal[t] = self._fill_nodal_array_1d(self.GFlexible[t], self.flexible_nodes)

        precharging_mask = (self.Storage[t] + 1e-6 < self.CPHS) & (precharge_energy > 1e-6) & precharging_mask 
        
        # Internode precharging
        if (Fillt.sum() > 1e-6) and (Surplust.sum() > 1e-6):
            self.Transmission[t] = get_transmission_flows_t(
                Fillt, Surplust, self.CTrans, self.network, self.networksteps,
                np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
            )

            Netloadt = Transmissiont_pre - self.Transmission[t].sum(axis=0)

            for i in range(len(self.flexible_nodes)):
                fidx = self.flexible_sorted[i]
                
                if flexible_trickling_mask[fidx]:
                    node = self.flexible_nodes[fidx]

                    change_power = max(min(Netloadt[node],trickling_change_max[fidx]),0)
                    change_energy = -change_power / self.resolution

                    self.GFlexible[t,fidx] += change_power
                    Netloadt[node] -= change_power
                    flexible_reserves[fidx] -= change_energy 

            for i in range(len(self.storage_nodes)):
                if precharging_mask[i]:
                    node = self.storage_nodes[i]

                    change_power = min(max(Netloadt[node], -precharging_change_max[i]),0)
                    change_energy = self._get_energy_change(self.SPower[t,i], change_power, i)

                    self.SPower[t,i] += change_power
                    Netloadt[node] -= change_power
                    precharge_energy[i] -= change_energy 

            self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)
            self.GFlexible_nodal[t] = self._fill_nodal_array_1d(self.GFlexible[t], self.flexible_nodes)

        return precharge_energy, flexible_reserves, flexible_trickling_mask
    
    def _determine_precharge_storage(self, t_precharge_start, t_end_deficit):
        for t in range(t_precharge_start, t_end_deficit):
            Storaget_p_lb = self.Storage[t-1] * self.storage_d_efficiencies / self.resolution 
            Storaget_p_ub = (self.CPHS - self.Storage[t-1]) / self.storage_c_efficiencies / self.resolution 
            self._Discharget_max = np.minimum(self.CPHP, Storaget_p_lb)
            self._Charget_max = np.minimum(self.CPHP, Storaget_p_ub)
            SPowert = self.SPower[t].copy()
            GFlexiblet = self.GFlexible[t].copy()

            if t in self.year_first_t:
                for i in range(len(self.year_first_t)):
                    if t == self.year_first_t[i]:
                        Flexiblet_p_lb = self.Flexible_Limits_Annual[i] / self.resolution
            else:
                Flexiblet_p_lb = self.GFlexible_constraint[t-1] / self.resolution 
            self._Flexible_max = np.minimum(self.CFlexible, Flexiblet_p_lb)

            self.SPower[t] = (
                    np.maximum(np.minimum(SPowert, self._Discharget_max),0) +
                    np.minimum(np.maximum(SPowert, -self._Charget_max),0) 
                    
                )    
            self.GFlexible[t] = np.minimum(GFlexiblet, self._Flexible_max)

            self.Storage[t] = self._update_storage(t, self.Storage[t-1])
            if t in self.year_first_t:
                for i in range(len(self.year_first_t)):
                    if t == self.year_first_t[i]:
                        self.GFlexible_constraint[t] = self.Flexible_Limits_Annual[i] - self.GFlexible[t] * self.resolution
            else:
                self.GFlexible_constraint[t] = self.GFlexible_constraint[t-1] - self.GFlexible[t] * self.resolution
        return None

    def _precharge_storage(self, t, Netload):
        trickling_reserves, precharge_energy, flexible_reserves, t_deficit_start = self._determine_precharge_energies(t, Netload)

        t_precharge_start = self._determine_precharge_powers(trickling_reserves, precharge_energy, flexible_reserves, t_deficit_start, Netload)

        self._determine_precharge_storage(t_precharge_start, t+1)

        return
    
    def _transmission_for_period(self, start_t, end_t, Netload, precharging_allowed):
        perform_precharge = False

        for t in range(start_t, end_t):
            """ if t%100 == 0:
                print(t) #### DEBUG """
            # Initialise time interval
            Storaget_p_lb = self.Storage[t-1] * self.storage_d_efficiencies / self.resolution 
            Storaget_p_ub = (self.CPHS - self.Storage[t-1]) / self.storage_c_efficiencies / self.resolution 
            self._Discharget_max = np.minimum(self.CPHP, Storaget_p_lb)
            self._Discharget_max_nodal = self._fill_nodal_array_1d(self._Discharget_max, self.storage_nodes)
            self._Charget_max = np.minimum(self.CPHP, Storaget_p_ub)
            self._Charget_max_nodal = self._fill_nodal_array_1d(self._Charget_max, self.storage_nodes)
            
            if t in self.year_first_t:
                for i in range(len(self.year_first_t)):
                    if t == self.year_first_t[i]:
                        Flexiblet_p_lb = self.Flexible_Limits_Annual[i] / self.resolution
            else:
                Flexiblet_p_lb = self.GFlexible_constraint[t-1] / self.resolution
            
            self._Flexible_max = np.minimum(self.CFlexible, Flexiblet_p_lb)
            self._Flexible_max_nodal = self._fill_nodal_array_1d(self._Flexible_max, self.flexible_nodes)

            Netloadt = Netload[t].copy()  
            if not precharging_allowed:
                self.Transmission[t] = np.zeros((len(self.CTrans), self.nodes), dtype = np.float64) 

            # Avoiding extra memory allocation. This will be overwritten after loop
            self.Deficit_nodal[t] = np.maximum(Netloadt, 0)
            
            if self.Deficit_nodal[t].sum() > 1e-6:
                # Fill deficits with transmitted spillage               
                self.Spillage_nodal[t] = (
                    -1 * np.minimum(0, Netloadt)
                )

                self.Transmission[t] = get_transmission_flows_t(
                    self.Deficit_nodal[t], self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt -= self.Transmission[t].sum(axis=0)

            self.Deficit_nodal[t] = np.maximum(Netloadt, 0)
            if self.Deficit_nodal[t].sum() > 1e-6:
                # Draw down from neighbours storage and flexible reserves                
                self.Spillage_nodal[t] = (
                    self._Discharget_max_nodal
                    + self._Flexible_max_nodal
                )

                self.Transmission[t] = get_transmission_flows_t(
                    self.Deficit_nodal[t], self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                self.SPower_nodal[t] = (
                    np.maximum(np.minimum(Netloadt, self._Discharget_max_nodal),0) +
                    np.minimum(np.maximum(Netloadt, -self._Charget_max_nodal),0) 
                    
                ) 

                self.GFlexible_nodal[t] = np.minimum(
                    np.maximum(Netloadt - self.SPower_nodal[t], 0),
                    self._Flexible_max_nodal
                ) 

            self.Spillage_nodal[t] = -1 * np.minimum(0, Netloadt - np.minimum(self.SPower_nodal[t], 0))
            """ if t>7:
                print(Netloadt)
                print(self.SPower_nodal[t])
                print(self.GFlexible_nodal[t])
                print(self.GFlexible[t]) """

            if self.Spillage_nodal[t].sum() > 1e-6:
                self.Transmission[t] = get_transmission_flows_t(
                    (self.SPower_nodal[t] + self._Charget_max_nodal), 
                    self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                self.SPower_nodal[t] = (
                    np.maximum(np.minimum(Netloadt, self._Discharget_max_nodal),0) +
                    np.minimum(np.maximum(Netloadt, -self._Charget_max_nodal),0) 
                ) 

                self.GFlexible_nodal[t] = np.minimum(
                    np.maximum(Netloadt - self.SPower_nodal[t], 0),
                    self._Flexible_max_nodal
                ) 
            
            # Apportion to individual storages/flexible 
            self.Deficit_nodal[t] = np.maximum(Netloadt - self.SPower_nodal[t] - self.GFlexible_nodal[t], 0) 
            """ if t>7:
                print(self.GFlexible_nodal[t])
                print(self.GFlexible[t]) """
            
            for node in range(self.nodes):
                # Apportion storage
                storage_mask = self.storage_nodes == node
                if np.any(storage_mask):
                    for idx in self.storage_sorted_nodal[node,:]:
                        if idx == -1:
                            break
                        self._clamp_and_assign(t, node, self.storage_order[storage_mask][idx])

                # Apportion flexible
                flexible_mask = self.flexible_nodes == node
                if np.any(flexible_mask):
                    for idx in self.flexible_sorted_nodal[node,:]: 
                        if idx == -1:
                            break 
                        self._clamp_and_assign(t, node, self.flexible_order[flexible_mask][idx], True) 

            """ if t>7:
                print(self.SPower_nodal[t])
                exit() """
            self.Storage[t] = self._update_storage(t, self.Storage[t-1])

            if t in self.year_first_t:
                for i in range(len(self.year_first_t)):
                    if t == self.year_first_t[i]:
                        self.GFlexible_constraint[t] = self.Flexible_Limits_Annual[i] - self.GFlexible[t] * self.resolution
            else:
                self.GFlexible_constraint[t] = self.GFlexible_constraint[t-1] - self.GFlexible[t] * self.resolution

            if not precharging_allowed:   
                continue

            if not perform_precharge and (self.Deficit_nodal[t].sum() > 1e-6):
                perform_precharge = True

            if (perform_precharge and (self.Deficit_nodal[t].sum() < 1e-6)):       
                self._precharge_storage(t, Netload)
                perform_precharge = False

    def _check_feasibility(self, Netload):
        # Check max_surplus is greater than 0 and less than 50% total energy
        max_surplus = -1 * np.minimum(0,Netload).sum() + min(self.Flexible_Limits_Annual.sum(),self.CFlexible.sum()*self.intervals) - np.maximum(0,Netload).sum()
        return (max_surplus > 0) and (max_surplus < self.energy*self.years*0.5)
    
    def _transmission_balancing(self):
        self.Storage[-1] = 0.5*self.CPHS
        
        Netload = (self.MLoad - self.GPV_nodal - self.GWind_nodal - self.GBaseload_nodal)

        # Ignore precharging unless likely to be close to optimal solution
        precharging_allowed = self._check_feasibility(Netload)

        self._transmission_for_period(0,self.intervals, Netload, precharging_allowed)   
        
        ImpExp = self.Transmission.sum(axis=1)   

        self.SPower_nodal = self._fill_nodal_array_2d(self.SPower, self.storage_nodes)
        self.Deficit_nodal = np.maximum(0, Netload - ImpExp - self._fill_nodal_array_2d(self.GFlexible, self.flexible_nodes) - self.SPower_nodal)
        self.Spillage_nodal = -1 * np.minimum(0, Netload - ImpExp - self.SPower_nodal) 
        self.TFlows = (self.trans_tflows_mask*self.Transmission).sum(axis=2)
        self.GDischarge = np.maximum(self.SPower, 0)

        return self.Deficit_nodal, np.abs(self.TFlows)

    def _objective(self) -> List[float]:
        """ start_time = time.time() """

        deficit, TFlowsAbs = self._transmission_balancing()
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution / self.years - self.allowance) * 1000000

        """ end_time = time.time()
        print(f"Transmission time: {end_time-start_time:.4f} seconds") """

        self._calculate_annual_generation()
        cost, _, _, _ = calculate_costs(self)

        loss = TFlowsAbs.sum(axis=0) * self.TLoss
        self.loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - self.loss) / 1000 # $/MWh
        
        """ print("LCOE: ", lcoe, pen_deficit, deficit.sum() / self.MLoad.sum(), self.GFlexible_annual)
        #exit() """
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
                    TSBaseload,
                    network,
                    transmission_mask,
                    intervals,
                    nodes,
                    lines,
                    years,
                    resolution,
                    allowance,
                    generator_ids,
                    generator_costs,
                    storage_ids,
                    storage_nodes,
                    flexible_ids,
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
                    solar_nodes,
                    wind_nodes,
                    flexible_nodes,
                    baseload_nodes,
                    CBaseload,
                    pv_cost_ids,
                    wind_cost_ids,
                    flexible_cost_ids,
                    baseload_cost_ids,
                    storage_cost_ids,
                    line_cost_ids,
                    networksteps,
                    storage_d_efficiencies,
                    storage_c_efficiencies,
                    Flexible_Limits_Annual,
                    first_year,
                    generator_line_ids,
                    storage_line_ids):
    result = np.empty(xs.shape[1], dtype=np.float64)
    for i in prange(xs.shape[1]):
        result[i] = objective_st(xs[:,i], 
                                MLoad,
                                TSPV,
                                TSWind,
                                TSBaseload,
                                network,
                                transmission_mask,
                                intervals,
                                nodes,
                                lines,
                                years,
                                resolution,
                                allowance,
                                generator_ids,
                                generator_costs,
                                storage_ids,
                                storage_nodes,
                                flexible_ids,
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
                                solar_nodes,
                                wind_nodes,
                                flexible_nodes,
                                baseload_nodes,
                                CBaseload,
                                pv_cost_ids,
                                wind_cost_ids,
                                flexible_cost_ids,
                                baseload_cost_ids,
                                storage_cost_ids,
                                line_cost_ids,
                                networksteps,
                                storage_d_efficiencies,
                                storage_c_efficiencies,
                                Flexible_Limits_Annual,
                                first_year,
                                generator_line_ids,
                                storage_line_ids)
    return result

@njit
def objective_st(x, 
                MLoad,
                TSPV,
                TSWind,
                TSBaseload,
                network,
                transmission_mask,
                intervals,
                nodes,
                lines,
                years,
                resolution,
                allowance,
                generator_ids,
                generator_costs,
                storage_ids,
                storage_nodes,
                flexible_ids,
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
                solar_nodes,
                wind_nodes,
                flexible_nodes,
                baseload_nodes,
                CBaseload,
                pv_cost_ids,
                wind_cost_ids,
                flexible_cost_ids,
                baseload_cost_ids,
                storage_cost_ids,
                line_cost_ids,
                networksteps,
                storage_d_efficiencies,
                storage_c_efficiencies,
                Flexible_Limits_Annual,
                first_year,
                generator_line_ids,
                storage_line_ids):
    solution = Solution_SingleTime(x,
                                MLoad,
                                TSPV,
                                TSWind,
                                TSBaseload,
                                network,
                                transmission_mask,
                                intervals,
                                nodes,
                                lines,
                                years,
                                resolution,
                                allowance,
                                generator_ids,
                                generator_costs,
                                storage_ids,
                                storage_nodes,
                                flexible_ids,
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
                                solar_nodes,
                                wind_nodes,
                                flexible_nodes,
                                baseload_nodes,
                                CBaseload,
                                pv_cost_ids,
                                wind_cost_ids,
                                flexible_cost_ids,
                                baseload_cost_ids,
                                storage_cost_ids,
                                line_cost_ids,
                                networksteps,
                                storage_d_efficiencies,
                                storage_c_efficiencies,
                                Flexible_Limits_Annual,
                                first_year,
                                generator_line_ids,
                                storage_line_ids)
    solution.evaluate()
    return solution.lcoe + solution.penalties