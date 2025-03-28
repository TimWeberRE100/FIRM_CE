import numpy as np
from typing import List
import time

from firm_ce.network import get_transmission_flows_t
from firm_ce.constants import JIT_ENABLED, EPSILON_FLOAT64, NP_FLOAT_MAX
from firm_ce.components.costs import calculate_costs
import firm_ce.helpers as helpers

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

        # Generators
        ('generator_ids', int64[:]),
        ('generator_costs', float64[:, :]),

        ('CPeak', float64[:]),
        ('CBaseload', float64[:]),
        ('GBaseload', float64[:, :]),
        ('GFlexible', float64[:, :]),

        # Storages
        ('storage_ids', int64[:]),
        ('storage_nodes', int64[:]),
        ('GDischarge', float64[:, :]),

        # Balancing
        ('flexible_ids', int64[:]),
        ('nodes_with_balancing', int64[:]), 
        ('storage_durations', float64[:]),
        ('storage_costs', float64[:, :]),        
        ('storage_d_efficiencies', float64[:]),
        ('storage_c_efficiencies', float64[:]),
        ('Storage', float64[:, :]),

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
        ('balancing_W_x', float64[:]),

        # Transmission
        ('Transmission', float64[:, :, :]),

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
        ('CPeak_nodal', float64[:]),
        ('CBaseload_nodal', float64[:]),
        ('GBaseload_nodal', float64[:, :]),
        ('Spillage_nodal', float64[:, :]),
        ('NetBalancing_nodal', float64[:, :]),
        ('Storage', float64[:, :]),
        ('Deficit_nodal', float64[:, :]),

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

        ('balancing_W_x_nodal', float64[:, :]),
        ('balancing_W_cutoffs', float64[:, :]),
        ('nodal_balancing_count', int64[:]),

        ('balancing_ids', int64[:]),
        ('balancing_nodes', int64[:]),
        ('balancing_order', int64[:]),
        ('balancing_storage_tag', boolean[:]),
        ('balancing_flexible_tag', boolean[:]),
        ('balancing_d_efficiencies', float64[:]),
        ('balancing_c_efficiencies', float64[:]),
        ('balancing_c_constraint', float64[:]),
        ('balancing_d_constraint', float64[:]),
        ('balancing_e_constraints', float64[:]),
        ('balancing_costs', float64[:, :]),
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
                generator_costs,
                storage_ids,
                storage_nodes,
                flexible_ids,
                nodes_with_balancing,
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

        self.MLoad = MLoad / 1000 # MW to GW
        self.intervals = intervals
        self.nodes = nodes
        self.lines = lines
        self.efficiency = efficiency
        self.resolution = resolution
        self.years = years
        self.energy = self.MLoad.sum() * self.resolution / self.years
        self.allowance = allowance*self.energy

        # Generators
        self.generator_ids = generator_ids
        self.generator_costs = generator_costs

        self.CPeak = CPeak
        self.CBaseload = CBaseload
        self.GBaseload = self.CBaseload * np.ones((self.intervals, len(self.CBaseload)), dtype=np.float64)
        self.GFlexible = np.zeros((self.intervals, len(self.CPeak)), dtype=np.float64)

        # Storages
        self.storage_ids = storage_ids
        self.storage_nodes = storage_nodes
        self.storage_durations = storage_durations
        self.storage_costs = storage_costs
        self.storage_d_efficiencies = storage_d_efficiencies
        self.storage_c_efficiencies = storage_c_efficiencies

        self.Storage = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)
        self.SPower = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)

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

        for idx in range(len(storage_durations)):
            if storage_durations[idx] > 0:
                self.CPHS[idx] = self.CPHP[idx] * storage_durations[idx]

        """ print(self.CPV,self.CWind,self.CFlexible,self.CPHS,self.CTrans,self.balancing_W_x) """

        # Transmission
        self.Transmission = np.zeros((self.intervals, len(self.CTrans), self.nodes), dtype = np.float64)

        # Flexible
        self.flexible_ids = flexible_ids
        self.balancing_ids = np.hstack((storage_ids, flexible_ids))
        self.balancing_order = np.arange(len(self.balancing_ids), dtype=np.int64)
        self.balancing_storage_tag = np.hstack(
                                        (np.full(len(storage_ids), True, dtype=np.bool_),
                                        np.full(len(flexible_ids), False, dtype=np.bool_))
                                    )
        self.balancing_flexible_tag = np.hstack(
                                        (np.full(len(storage_ids), False, dtype=np.bool_),
                                        np.full(len(flexible_ids), True, dtype=np.bool_))
                                    )
        
        self.flexible_mask = helpers.isin_numba(np.arange(self.generator_costs.shape[1], dtype=np.int64), flexible_cost_ids)

        # Nodal Values
        self.flexible_nodes = flexible_nodes 
        self.baseload_nodes = baseload_nodes
        self.storage_nodes = self.storage_nodes

        self.CFlexible_nodal = self._fill_nodal_array_1d(self.CFlexible, self.flexible_nodes)
        self.CPHP_nodal = self._fill_nodal_array_1d(self.CPHP, self.storage_nodes)
        self.CPHS_nodal = self._fill_nodal_array_1d(self.CPHS, self.storage_nodes)    
        self.GPV = self.CPV[np.newaxis, :] * TSPV
        self.GPV_nodal = self._fill_nodal_array_2d(self.GPV, solar_nodes)
        self.GWind = self.CWind[np.newaxis, :] * TSWind
        self.GWind_nodal = self._fill_nodal_array_2d(self.GWind, wind_nodes)
        self.CPeak_nodal = self._fill_nodal_array_1d(self.CPeak, self.flexible_nodes)
        self.CBaseload_nodal = self._fill_nodal_array_1d(self.CBaseload, self.baseload_nodes)
        self.GBaseload_nodal = self._fill_nodal_array_2d(self.GBaseload, self.baseload_nodes)
        
        self.SPower_nodal=  np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.GFlexible_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Spillage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Deficit_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)  

        self.GPV_annual = np.zeros(self.CPV.shape, dtype=np.float64)
        self.GWind_annual = np.zeros(self.CWind.shape, dtype=np.float64)
        self.GFlexible_annual = np.zeros(self.CFlexible.shape, dtype=np.float64)
        self.GBaseload_annual = np.zeros(self.CBaseload.shape, dtype=np.float64)
        self.GDischarge_annual = np.zeros(self.CPHP.shape, dtype=np.float64)
        self.TFlowsAbs_annual = np.zeros(0, dtype=np.float64)

        # Cost Values
        self.pv_cost_ids = pv_cost_ids
        self.wind_cost_ids = wind_cost_ids
        self.flexible_cost_ids = flexible_cost_ids
        self.baseload_cost_ids = baseload_cost_ids
        self.storage_cost_ids = storage_cost_ids
        self.line_cost_ids = line_cost_ids

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

    def _calculate_costs(self):
        solution_cost = calculate_costs(self)
        return solution_cost
    
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

    def _calculate_annual_generation(self):
        self.GPV_annual = self.GPV.sum(axis=0) * self.resolution / self.years
        self.GWind_annual = self.GWind.sum(axis=0) * self.resolution / self.years
        self.GDischarge_annual = self.GDischarge.sum(axis=0) * self.resolution / self.years
        self.GFlexible_annual = self.GFlexible.sum(axis=0) * self.resolution / self.years
        self.GBaseload_annual = self.GBaseload.sum(axis=0) * self.resolution / self.years
        self.TFlowsAbs_annual = self.TFlowsAbs.sum(axis=0) * self.resolution / self.years

        return None
    
    def _determine_precharge_energies(self, t, Netload):
        storage_order = self.balancing_order[self.balancing_storage_tag]
        flexible_order = self.balancing_order[self.balancing_flexible_tag]
        F_variable_costs = self.generator_costs[3, self.flexible_mask] ##### ADD FUEL COSTS

        while True:
            t -= 1

            self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)
            self.GFlexible_nodal[t] = self._fill_nodal_array_1d(self.GFlexible[t], self.flexible_nodes)
            trickling_reserves = self.CPHS
            precharging_reserves = np.zeros_like(self.CPHS, dtype=np.float64)

            # If you reach end of deficit block, return results
            if ((Netload[t] - self.SPower_nodal[t] - self.GFlexible_nodal[t] - self.Transmission[t].sum(axis=0) < 1e-6).all()):
                precharge_energy = (self.Storage[t+1] - self.Storage[t]) / self.storage_c_efficiencies

                return trickling_reserves, precharging_reserves, precharge_energy, t+1 
            
            # Otherwise, perform reverse-time charging
            else:
                trickling_reserves = np.minimum(trickling_reserves, self.Storage[t])
                precharging_reserves = np.minimum(self.CPHS - self.Storage[t], precharging_reserves)

                Storaget_p_lb_rev = self.Storage[t] * self.storage_c_efficiencies / self.resolution 
                Storaget_p_ub_rev = (self.CPHS - self.Storage[t]) / self.storage_d_efficiencies / self.resolution
                Discharget_max = np.minimum(self.CPHP, Storaget_p_ub_rev) # Reversed energy constraint in reverse time
                Discharget_max_nodal = self._fill_nodal_array_1d(Discharget_max, self.storage_nodes)
                Charget_max = np.minimum(self.CPHP, Storaget_p_lb_rev) # Reversed energy constraint in reverse time
                Charget_max_nodal = self._fill_nodal_array_1d(Charget_max, self.storage_nodes)

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
                        Discharget_max_nodal
                        + self.CFlexible_nodal
                    )

                    self.Transmission[t] = get_transmission_flows_t(
                        self.Deficit_nodal[t], self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                        np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                    )

                    Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                    self.SPower_nodal[t] = (
                        np.maximum(np.minimum(Netloadt, Discharget_max_nodal),0) +
                        np.minimum(np.maximum(Netloadt, -Charget_max_nodal),0)                     
                    ) 

                    self.GFlexible_nodal[t] = np.minimum(
                        np.maximum(Netloadt - self.SPower_nodal[t], 0),
                        self.CFlexible_nodal
                    ) 

                self.Spillage_nodal[t] = -1 * np.minimum(0, Netloadt - np.minimum(self.SPower_nodal[t], 0))
                if self.Spillage_nodal[t].sum() > 1e-6:
                    self.Transmission[t] = get_transmission_flows_t(
                        (self.SPower_nodal[t] + Charget_max_nodal), 
                        self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                        np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                    )

                    Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                    self.SPower_nodal[t] = (
                        np.maximum(np.minimum(Netloadt, Discharget_max_nodal),0) +
                        np.minimum(np.maximum(Netloadt, -Charget_max_nodal),0) 
                    ) 

                    self.GFlexible_nodal[t] = np.minimum(
                        np.maximum(Netloadt - self.SPower_nodal[t], 0),
                        self.CFlexible_nodal
                    ) 

                # Apportion to individual storages/flexible 
                self.Deficit_nodal[t] = np.maximum(Netloadt - self.SPower_nodal[t] - self.GFlexible_nodal[t], 0) 
                
                for node in range(self.nodes):
                    # Apportion storage
                    storage_mask = self.storage_nodes == node
                    if np.any(storage_mask):
                        sorted_indices = np.argsort((self.CPHS/self.CPHP)[storage_mask]) #### Move this out of loop
                        
                        L = sorted_indices.shape[0]
                        order_indices = np.empty(L, dtype=sorted_indices.dtype)
                        for i in range(L):
                            order_indices[i] = L - sorted_indices[i] - 1

                        for idx in order_indices:
                            storage_order_i = storage_order[storage_mask][idx]
                            
                            new_value = helpers.scalar_clamp(self.SPower_nodal[t][node], 
                                                            -Charget_max[storage_order_i], 
                                                            Discharget_max[storage_order_i])
                            
                            self.SPower_nodal[t][node] -= new_value
                            self.SPower[t][storage_order_i] = new_value

                    # Apportion flexible
                    flexible_mask = self.flexible_nodes == node
                    if np.any(flexible_mask):
                        sorted_indices = np.argsort(F_variable_costs[flexible_mask]) # Move this out of loop

                        L = sorted_indices.shape[0]
                        order_indices = np.empty(L, dtype=sorted_indices.dtype)
                        for i in range(L):
                            order_indices[i] = L - sorted_indices[i] - 1
                        
                        for idx in order_indices:                        
                            flexible_order_i = flexible_order[flexible_mask][idx] - len(storage_order)
                            
                            new_value = helpers.scalar_clamp(self.GFlexible_nodal[t][node], 
                                                            0, 
                                                            self.CFlexible[flexible_order_i])
                            self.GFlexible_nodal[t][node] -= new_value
                            self.GFlexible[t][flexible_order_i] = new_value
            
            self.Storage[t-1] = (self.Storage[t] 
                               + np.maximum(self.SPower[t], 0) * self.resolution / self.storage_d_efficiencies 
                               + np.minimum(self.SPower[t], 0) * self.resolution * self.storage_c_efficiencies) # Reverse charge/discharge effect in reverse time
    
    def _determine_precharge_powers(self, trickling_reserves, precharging_reserves, precharge_energy, t_first_deficit, Netload):
        keep_precharging = True
        while keep_precharging:
            t -= 1

            Storaget_p_lb = trickling_reserves * self.storage_d_efficiencies / self.resolution 
            Storaget_p_ub = precharging_reserves / self.storage_c_efficiencies / self.resolution 
            Discharget_max = np.minimum(self.CPHP, Storaget_p_lb)                
            Charget_max = np.minimum(self.CPHP, Storaget_p_ub)  

            trickling_mask = trickling_reserves > 1e-6
            precharging_mask = precharging_reserves > 1e-6

            Transmissiont_pre = self.Transmission[t].sum(axis=0)

            SPowert_trickling = self.SPower[t].copy()
            SPowert_trickling[~trickling_mask] = 0
            SPowert_charging = self.SPower[t].copy()
            SPowert_charging[~precharging_mask] = 0
            SPowert_trickling_nodal = self._fill_nodal_array_1d(SPowert_trickling, self.storage_nodes)
            SPowert_charging_nodal = self._fill_nodal_array_1d(SPowert_charging, self.storage_nodes)

            Discharget_max[~trickling_mask] = 0
            Discharget_max_nodal = self._fill_nodal_array_1d(Discharget_max, self.storage_nodes)
            Charget_max[~precharging_mask] = 0
            Charget_max_nodal = self._fill_nodal_array_1d(Charget_max, self.storage_nodes)
            Charget = np.maximum(Charget_max_nodal + SPowert_charging_nodal,0)
            Discharget = np.maximum(Discharget_max_nodal - SPowert_trickling_nodal, 0)  
                
            Fillt = np.maximum(Charget - Discharget,0)
            Surplust = -1 * np.minimum(Charget - Discharget,0)

            # Intranode precharging
            IntranodeCharget = np.minimum(Charget, Discharget)
            Intranodet_trickle = IntranodeCharget.copy()
            Intranodet_charge = -IntranodeCharget.copy()

            for i in range(len(self.storage_nodes)):
                ##### FLEXIBLE NEEDS TO BE ACCOUNTED FOR HERE
                node = self.storage_nodes[i]

                ####### Changes here need to properly account for efficienies like below. It is currently wrong
                if trickling_mask[i]:
                    change = np.maximum(np.minimum(Intranodet_trickle[node],Discharget_max[i] - self.SPower[t,i]),0)

                    if change < 0:
                        charge_energy_change = -1 * (self.SPower[t,i] + change) * self.resolution * self.storage_c_efficiencies[i]
                        discharge_energy_change = min(max(self.SPower[t,i],0), -change) * self.resolution / self.storage_d_efficiencies[i]
                    else:
                        charge_energy_change = -1 * max(min(self.SPower[t,i],0), change)
                        discharge_energy_change = (self.SPower[t,i] + change) * self.resolution / self.storage_d_efficiencies[i]

                    self.SPower[t,i] += change
                    Intranodet_trickle[node] -= change 
                    trickling_reserves[i] -= (discharge_energy_change + charge_energy_change)  
                                         
                if precharging_mask[i]:
                    change = np.minimum(np.maximum(Intranodet_charge[node],-Charget_max[i] - self.SPower[t,i]),0)

                    if change < 0:
                        charge_energy_change = -1 * (self.SPower[t,i] + change) * self.resolution * self.storage_c_efficiencies[i]
                        discharge_energy_change = min(max(self.SPower[t,i],0), -change) * self.resolution / self.storage_d_efficiencies[i]
                    else:
                        charge_energy_change = -1 * max(min(self.SPower[t,i],0), change)
                        discharge_energy_change = (self.SPower[t,i] + change) * self.resolution / self.storage_d_efficiencies[i]

                    self.SPower[t,i] += change
                    Intranodet_charge[node] -= change
                    precharging_reserves[i] -= (discharge_energy_change + charge_energy_change)  
                    precharge_energy[i] -= (discharge_energy_change + charge_energy_change) 

            self.SPower_nodal[t] = self._fill_nodal_array_1d(self.SPower[t], self.storage_nodes)
            
            # Internode precharging
            if (Fillt.sum() > 1e-6) and (Surplust.sum() > 1e-6):
                self.Transmission[t] = get_transmission_flows_t(
                    Fillt, Surplust, self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Transmissiont_pre - self.Transmission[t].sum(axis=0)

                for i in range(len(self.storage_nodes)):
                    ##### FLEXIBLE NEEDS TO BE ACCOUNTED FOR HERE
                    node = self.storage_nodes[i]

                    change = (
                        max(min(Netloadt[node], Discharget_max[i]),0) +
                        min(max(Netloadt, -Charget_max),0)
                    )

                    ###### This is wrong. It also depends on whether SPower starts positive or negative.
                    ######### Also ought to update based on the masks
                    if change < 0:
                        charge_energy_change = -1 * (self.SPower[t,i] + change) * self.resolution * self.storage_c_efficiencies[i]
                        discharge_energy_change = min(max(self.SPower[t,i],0), -change) * self.resolution / self.storage_d_efficiencies[i]
                    else:
                        charge_energy_change = -1 * max(min(self.SPower[t,i],0), change)
                        discharge_energy_change = (self.SPower[t,i] + change) * self.resolution / self.storage_d_efficiencies[i]

                    if trickling_mask[i]:
                        trickling_reserves[i] -= (discharge_energy_change + charge_energy_change)  
                    if precharging_mask[i]:
                        precharging_reserves[i] -= (discharge_energy_change + charge_energy_change)  
                        precharge_energy[i] -= (discharge_energy_change + charge_energy_change)               

                    self.SPower[t,i] += change
                    Netloadt[node] -= change 

            precharging_mask = precharging_mask & (precharging_reserves > 1e-6) & (precharge_energy > 1e-6)
            trickling_mask = trickling_mask & (trickling_reserves > 1e-6)

            if (precharge_energy < 1e-6) or (not precharging_mask.any()) or (not trickling_mask.any()):
                t_precharge_start = t
                break
                
        return t_precharge_start
    
    def _determine_precharge_energies(self, t_precharge_start, t_deficit_end):
        for t in range(t_precharge_start, t_deficit_end):
            self.Storage[t] = (self.Storage[t-1] 
                               - np.maximum(self.SPower[t], 0) / self.storage_d_efficiencies * self.resolution 
                               - np.minimum(self.SPower[t], 0) * self.storage_c_efficiencies * self.resolution)

        return None

    def _precharge_storage(self, t, Netload, deficit_precharging):
        trickling_reserves, precharging_reserves, precharge_energy, t_first_deficit = self._determine_precharge_energies(t, Netload)

        t_precharge_start = self._determine_precharge_powers(trickling_reserves, precharging_reserves, precharge_energy, t_first_deficit, Netload)

        self._determine_precharge_energies(t_precharge_start, t)

        return
    
    def _transmission_for_period(self, start_t, end_t, Netload, precharging_allowed):
        TEST_T = 381
        storage_order = self.balancing_order[self.balancing_storage_tag]
        flexible_order = self.balancing_order[self.balancing_flexible_tag]
        F_variable_costs = self.generator_costs[3, self.flexible_mask] ##### ADD FUEL COSTS        
        
        perform_precharge = False
        deficit_precharging = True

        for t in range(start_t, end_t):
            # Initialise time interval
            Storaget_p_lb = self.Storage[t-1] * self.storage_d_efficiencies / self.resolution 
            Storaget_p_ub = (self.CPHS - self.Storage[t-1]) / self.storage_c_efficiencies / self.resolution 
            Discharget_max = np.minimum(self.CPHP, Storaget_p_lb)
            Discharget_max_nodal = self._fill_nodal_array_1d(Discharget_max, self.storage_nodes)
            Charget_max = np.minimum(self.CPHP, Storaget_p_ub)
            Charget_max_nodal = self._fill_nodal_array_1d(Charget_max, self.storage_nodes)

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
                    Discharget_max_nodal
                    + self.CFlexible_nodal
                )

                self.Transmission[t] = get_transmission_flows_t(
                    self.Deficit_nodal[t], self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                self.SPower_nodal[t] = (
                    np.maximum(np.minimum(Netloadt, Discharget_max_nodal),0) +
                    np.minimum(np.maximum(Netloadt, -Charget_max_nodal),0) 
                    
                ) 

                self.GFlexible_nodal[t] = np.minimum(
                    np.maximum(Netloadt - self.SPower_nodal[t], 0),
                    self.CFlexible_nodal
                ) 

            self.Spillage_nodal[t] = -1 * np.minimum(0, Netloadt - np.minimum(self.SPower_nodal[t], 0))

            if self.Spillage_nodal[t].sum() > 1e-6:
                self.Transmission[t] = get_transmission_flows_t(
                    (self.SPower_nodal[t] + Charget_max_nodal), 
                    self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t]), np.minimum(0, self.Transmission[t])
                )

                Netloadt = Netload[t] - self.Transmission[t].sum(axis=0)

                self.SPower_nodal[t] = (
                    np.maximum(np.minimum(Netloadt, Discharget_max_nodal),0) +
                    np.minimum(np.maximum(Netloadt, -Charget_max_nodal),0) 
                ) 

                self.GFlexible_nodal[t] = np.minimum(
                    np.maximum(Netloadt - self.SPower_nodal[t], 0),
                    self.CFlexible_nodal
                ) 
            
            # Apportion to individual storages/flexible 
            self.Deficit_nodal[t] = np.maximum(Netloadt - self.SPower_nodal[t] - self.GFlexible_nodal[t], 0) 
            
            for node in range(self.nodes):
                # Apportion storage
                storage_mask = self.storage_nodes == node
                if np.any(storage_mask):
                    sorted_indices = np.argsort((self.CPHS/self.CPHP)[storage_mask]) #### Move this out of loop
                    
                    L = sorted_indices.shape[0]
                    order_indices = np.empty(L, dtype=sorted_indices.dtype)
                    for i in range(L):
                        order_indices[i] = sorted_indices[i]

                    for idx in order_indices:
                        storage_order_i = storage_order[storage_mask][idx]
                        
                        new_value = helpers.scalar_clamp(self.SPower_nodal[t][node], 
                                                         -Charget_max[storage_order_i], 
                                                         Discharget_max[storage_order_i])
                        
                        self.SPower_nodal[t][node] -= new_value
                        self.SPower[t][storage_order_i] = new_value

                # Apportion flexible
                flexible_mask = self.flexible_nodes == node
                if np.any(flexible_mask):
                    sorted_indices = np.argsort(F_variable_costs[flexible_mask]) # Move this out of loop

                    L = sorted_indices.shape[0]
                    order_indices = np.empty(L, dtype=sorted_indices.dtype)
                    for i in range(L):
                        order_indices[i] = L - sorted_indices[i] - 1
                    
                    for idx in order_indices:                        
                        flexible_order_i = flexible_order[flexible_mask][idx] - len(storage_order)
                        
                        new_value = helpers.scalar_clamp(self.GFlexible_nodal[t][node], 
                                                         0, 
                                                         self.CFlexible[flexible_order_i])
                        self.GFlexible_nodal[t][node] -= new_value
                        self.GFlexible[t][flexible_order_i] = new_value
            
            self.Storage[t] = (self.Storage[t-1] 
                               - np.maximum(self.SPower[t], 0) / self.storage_d_efficiencies * self.resolution 
                               - np.minimum(self.SPower[t], 0) * self.storage_c_efficiencies * self.resolution)

            if not perform_precharge and (self.Deficit_nodal[t].sum() > 1e-6):
                perform_precharge = True
                deficit_precharging = True

            if (perform_precharge and (self.Deficit_nodal[t].sum() < 1e-6)):
                self._precharge_storage(t, Netload, deficit_precharging)
                break

    def _check_feasibility(self, Netload):
        # Check max_surplus is greater than 0 and less than 50% total energy
        #### This may not be a good check if Flexible starting capacity is very large
        max_surplus = -1 * np.minimum(0,Netload).sum() + self.CFlexible.sum()*self.intervals - np.maximum(0,Netload).sum()
        print(f"MAX SURPLUS: {max_surplus} {max_surplus / (self.energy*self.years)}")
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
        self.TFlows = (self.Transmission).sum(axis=2)
        self.GDischarge = np.maximum(self.SPower, 0)

        np.savetxt("results/Netload.csv", Netload, delimiter=",")
        np.savetxt("results/ImpExp.csv", ImpExp, delimiter=",")
        np.savetxt("results/Deficit.csv", self.Deficit_nodal, delimiter=",")
        np.savetxt("results/Spillage.csv", self.Spillage_nodal, delimiter=",")
        np.savetxt("results/Storage.csv", self.Storage, delimiter=",")
        np.savetxt("results/SPower.csv", self.SPower, delimiter=",")
        np.savetxt("results/Flexible.csv", self.GFlexible, delimiter=",")
        #OUTPUT = np.vstack((Netload[:18000,0],ImpExp[:18000,0],SPower[:18000,0] - storage_p_profiles[:18000,0],SPower[:18000,5] - storage_p_profiles[:18000,5],Flexible[:18000,0] - flexible_p_profiles[:18000,0],self.Deficit_nodal[:18000,0],self.Spillage_nodal[:18000,0],self.Storage[:18000,0],self.Storage[:18000,5],SPower[:18000,0],SPower[:18000,5],Flexible[:18000,0]))
        OUTPUT = np.concatenate((self.MLoad[:18000,:],self.GPV[:18000,:],self.GWind[:18000,:],self.GBaseload[:18000,:],self.GFlexible[:18000,:],self.SPower[:18000,:],self.Deficit_nodal[:18000,:],self.Spillage_nodal[:18000,:],self.Storage[:18000,:],ImpExp[:18000,:]), axis=1)
        OUTPUT2 = np.concatenate((Netload[:18000,:],ImpExp[:18000,:],(self.GFlexible)[:18000,:],(self.SPower)[:18000,:],self.Deficit_nodal[:18000,:],self.Spillage_nodal[:18000,:],self.Storage[:18000,:]), axis=1)
        
        np.savetxt("results/OUTPUT.csv", 1000*OUTPUT, delimiter=",")
        np.savetxt("results/OUTPUT2.csv", 1000*OUTPUT2, delimiter=",")

        return self.Deficit_nodal, np.abs(self.TFlows)

    def _objective(self) -> List[float]:
        start_time = time.time()

        deficit, TFlowsAbs = self._transmission_balancing()
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution / self.years - self.allowance) * 1000000

        end_time = time.time()
        print(f"Transmission time: {end_time-start_time:.4f} seconds")

        self._calculate_annual_generation()
        cost = self._calculate_costs()

        loss = TFlowsAbs.sum(axis=0) * self.TLoss
        loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - loss) / 1000 # $/MWh
        
        print("LCOE: ", lcoe, pen_deficit, deficit.sum() / self.MLoad.sum())
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
                    generator_costs,
                    storage_ids,
                    storage_nodes,
                    flexible_ids,
                    nodes_with_balancing,
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
                                generator_costs,
                                storage_ids,
                                storage_nodes,
                                flexible_ids,
                                nodes_with_balancing,
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
                generator_costs,
                storage_ids,
                storage_nodes,
                flexible_ids,
                nodes_with_balancing,
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
                                generator_costs,
                                storage_ids,
                                storage_nodes,
                                flexible_ids,
                                nodes_with_balancing,
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