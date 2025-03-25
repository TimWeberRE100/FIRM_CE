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
    
    def _precharge_storage(self, t_pre_end, t, precharge_intervals, precharge_mask, Netload, deficit_precharging):
        TEST_T = 406 ####### DEBUG
        storage_cycle_orders = np.array([1,1,1,1,1,0,0,0,0,0], dtype=np.int32) ######## DEBUG
        frequency_orders = np.array([0,0,0,0]) ##### DEBUG

        t_pre_initial = max(t_pre_end - max(precharge_intervals),0)
        max_cycle_order = max(storage_cycle_orders)

        for t_pre in range(t_pre_initial, t_pre_end):
            self._transmission_for_period(t_pre, t_pre+1, Netload, False) # Run normal transmission for interval first

            Storaget_p_lb = self.Storage[t_pre-1] / (self.resolution * self.storage_d_efficiencies)
            Storaget_p_ub = (self.CPHS - self.Storage[t_pre-1]) / (self.resolution * self.storage_c_efficiencies)
            Discharget_max = np.minimum(self.CPHP, Storaget_p_lb)                
            Charget_max = np.minimum(self.CPHP, Storaget_p_ub)  

            # Charge longest-duration storage with flexible in cost order
            if deficit_precharging: # Skip flexible precharging if attempting to capture future spillage
                for i in range(max(frequency_orders) + 1):
                    storage_precharging = (storage_cycle_orders == max_cycle_order) & precharge_mask
                    SPowert_precharge = np.zeros(len(self.storage_ids), dtype=np.float64)
                    GFlexible_precharge = np.zeros(len(self.flexible_ids), dtype=np.float64)                              

                    SPowert_charging = self.SPower[t_pre].copy()
                    SPowert_charging[~precharge_mask] = 0
                    SPowert_charging_nodal = self._fill_nodal_array_1d(SPowert_charging, self.storage_nodes)
                    GFlexible_trickle = self.GFlexible[t_pre].copy()

                    Charget_max[~precharge_mask] = 0
                    Charget_max_nodal = self._fill_nodal_array_1d(Charget_max, self.storage_nodes)                

                    Fillt = np.maximum(Charget_max_nodal + SPowert_charging_nodal,0)
                    Surplust = np.maximum(self.CFlexible_nodal - self._fill_nodal_array_1d(GFlexible_trickle, self.flexible_nodes), 0)  
                    Transmissiont_pre = self.Transmission[t_pre].sum(axis=0)

                    self.Transmission[t_pre] = get_transmission_flows_t(
                        Fillt, Surplust, self.CTrans, self.network, self.networksteps,
                        np.maximum(0, self.Transmission[t_pre]), np.minimum(0, self.Transmission[t_pre])
                    )

                    Netloadt = Transmissiont_pre - self.Transmission[t_pre].sum(axis=0)

                    for node in range(self.nodes):
                        flexible_node_mask = (self.flexible_nodes == node)
                        precharge_node_mask = (precharge_mask & (self.storage_nodes == node))

                        #### Might not able to perfectly split the surplus because of constraints
                        if flexible_node_mask.sum() > 0:
                            GFlexible_precharge[flexible_node_mask] = np.minimum(
                                np.full(flexible_node_mask.sum(), max(Netloadt[node],0) / flexible_node_mask.sum(), dtype=np.float64), 
                                np.maximum(self.CFlexible[flexible_node_mask] - GFlexible_trickle[flexible_node_mask],0)
                                ) 
                        if precharge_node_mask.sum() > 0:
                            SPowert_precharge[precharge_node_mask] =  np.maximum(
                                np.full(precharge_node_mask.sum(), min(Netloadt[node],0) / precharge_node_mask.sum()), 
                                -1*np.maximum(Charget_max[precharge_node_mask] + SPowert_charging[precharge_node_mask],0)
                                )     ### Add the discharge/charge constraint opposites to each one?               

                    self.GFlexible[t_pre] += GFlexible_precharge 
                    self.SPower[t_pre] += SPowert_precharge

            # Charge storage based upon cycle frequency order            
            for i in range(max_cycle_order):                 
                if precharge_intervals[i] < t_pre_end - t_pre:
                    continue

                # For spillage precharging, reverse the order
                if deficit_precharging:
                    storage_precharging = (storage_cycle_orders == i) & precharge_mask
                    storage_trickling = storage_cycle_orders == i+1  
                else:
                    storage_precharging = (storage_cycle_orders == i+1) & precharge_mask
                    storage_trickling = storage_cycle_orders == i

                SPowert_precharge = np.zeros(len(self.storage_ids), dtype=np.float64)
                storage_trickling = storage_cycle_orders == i+1  

                Transmissiont_pre = self.Transmission[t_pre].sum(axis=0)

                SPowert_trickling = self.SPower[t_pre].copy()
                SPowert_trickling[~storage_trickling] = 0
                SPowert_charging = self.SPower[t_pre].copy()
                SPowert_charging[~storage_precharging] = 0
                SPowert_trickling_nodal = self._fill_nodal_array_1d(SPowert_trickling, self.storage_nodes)
                SPowert_charging_nodal = self._fill_nodal_array_1d(SPowert_charging, self.storage_nodes)

                Discharget_max[~storage_trickling] = 0
                Discharget_max_nodal = self._fill_nodal_array_1d(Discharget_max, self.storage_nodes)
                Charget_max[~storage_precharging] = 0
                Charget_max_nodal = self._fill_nodal_array_1d(Charget_max, self.storage_nodes)
                Fillt = np.maximum(Charget_max_nodal + SPowert_charging_nodal,0)
                Surplust = np.maximum(Discharget_max_nodal - SPowert_trickling_nodal, 0) #-1 * np.minimum(Netloadt - np.minimum(self.SPower_nodal[t_pre], 0), 0) 

                if t_pre == TEST_T:
                    print(f'-----{t_pre} pre1------')
                    print(SPowert_trickling)
                    print(SPowert_charging)   
                    print(Netloadt)             
                    print(Fillt) 
                    print(Surplust) 
                    print('-------------')
                    print(self.Storage[t_pre-1])     

                    print(f'-----{t_pre} pre2------')
                    print(t_pre)
                    print(Storaget_p_lb)
                    print(Storaget_p_ub)
                    print(Discharget_max)
                    print(Charget_max)
                    print(Discharget_max_nodal)
                    print(Charget_max_nodal)
                        
                #print(self.Storage[t_pre]) 

                self.Transmission[t_pre] = get_transmission_flows_t(
                    Fillt, Surplust, self.CTrans, self.network, self.networksteps,
                    np.maximum(0, self.Transmission[t_pre]), np.minimum(0, self.Transmission[t_pre])
                )

                Netloadt = Transmissiont_pre - self.Transmission[t_pre].sum(axis=0)
                        
                for node in range(self.nodes):
                    trickle_node_mask = (storage_trickling & (self.storage_nodes == node))
                    precharge_node_mask = (storage_precharging & (self.storage_nodes == node))

                    #### Might not able to perfectly split the surplus because of constraints
                    if trickle_node_mask.sum() > 0:
                        SPowert_precharge[trickle_node_mask] = np.minimum(
                            np.full(trickle_node_mask.sum(), max(Netloadt[node],0) / trickle_node_mask.sum(), dtype=np.float64), 
                            np.maximum(Discharget_max[trickle_node_mask] - SPowert_trickling[trickle_node_mask],0)
                            ) 
                    if precharge_node_mask.sum() > 0:
                        SPowert_precharge[precharge_node_mask] =  np.maximum(
                            np.full(precharge_node_mask.sum(), min(Netloadt[node],0) / precharge_node_mask.sum()), 
                            -1*np.maximum(Charget_max[precharge_node_mask] + SPowert_charging[precharge_node_mask],0)
                            )     ### Add the discharge/charge constraint opposites to each one?               

                self.SPower[t_pre] += SPowert_precharge  

                if t_pre == TEST_T:
                    print(f'-----{t_pre} pre3------')  
                    print(Netloadt)             
                    print(self.SPower[t_pre])      
                    print(self.Transmission[t_pre].sum(axis=0))                

            self.Storage[t_pre] = (self.Storage[t_pre-1] 
                       - np.maximum(self.SPower[t_pre], 0) * self.storage_d_efficiencies * self.resolution 
                        - np.minimum(self.SPower[t_pre], 0) * self.storage_c_efficiencies * self.resolution)        

        # Perform normal dispatch of deficit period
        self._transmission_for_period(t_pre_end, t, Netload, False)
        
        return None
    
    def _transmission_for_period(self, start_t, end_t, Netload, precharging_allowed):
        TEST_T = 406
        storage_order = self.balancing_order[self.balancing_storage_tag]
        flexible_order = self.balancing_order[self.balancing_flexible_tag]
        F_variable_costs = self.generator_costs[3, self.flexible_mask] ##### ADD FUEL COSTS        
        
        precharge_energy = 0
        perform_precharge = False
        deficit_precharging = True
        precharge_mask = np.full(len(self.storage_ids), False, dtype=np.bool_)
        t_pre_end = 0

        for t in range(start_t, end_t):
            # Perform precharging
            if precharging_allowed and (precharge_energy > 0) and (perform_precharge): 
                # precharge_intervals = self._determine_precharge_intervals() #### REMEMBER CHARGING EFFICIENCIES
                precharge_intervals = np.array([4,160], dtype=np.int64) ########## DEBUG
                self._precharge_storage(t_pre_end, t, precharge_intervals, precharge_mask, Netload, deficit_precharging)  
                precharge_mask = np.full(len(self.storage_ids), False, dtype=np.bool_)           
                precharge_energy = 0   
            elif not precharging_allowed:
                self.Transmission[t] = np.zeros((len(self.CTrans), self.nodes),dtype=np.float64)
                self.SPower[t] = np.zeros(len(self.storage_ids), dtype=np.float64)
                self.GFlexible[t] = np.zeros(len(self.flexible_ids), dtype=np.float64)

            # Initialise time interval
            Storaget_p_lb = self.Storage[t-1] / (self.resolution * self.storage_d_efficiencies)
            Storaget_p_ub = (self.CPHS - self.Storage[t-1]) / (self.resolution * self.storage_c_efficiencies)
            Discharget_max = np.minimum(self.CPHP, Storaget_p_lb)
            Discharget_max_nodal = self._fill_nodal_array_1d(Discharget_max, self.storage_nodes)
            Charget_max = np.minimum(self.CPHP, Storaget_p_ub)
            Charget_max_nodal = self._fill_nodal_array_1d(Charget_max, self.storage_nodes)

            Netloadt = Netload[t].copy()  

            # Avoiding extra memory allocation. This will be overwritten after loop
            self.Deficit_nodal[t] = np.maximum(Netloadt, 0)

            if t == TEST_T:
                print('---------1---------')
                print(t, precharging_allowed)
                print(Netloadt)
                print(self.SPower_nodal[t])
                print(self.GFlexible_nodal[t])
                print(self.Deficit_nodal[t])
                print(self.Transmission[t].sum(axis=0))
            
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

                """ self.SPower_nodal[t] = (
                    np.maximum(np.minimum(Netloadt, Discharget_max_nodal),0) +
                    np.minimum(np.maximum(Netloadt, -Charget_max_nodal),0) 
                    
                ) 

                self.GFlexible_nodal[t] = np.minimum(
                    np.maximum(Netloadt - self.SPower_nodal[t], 0),
                    self.CFlexible_nodal
                )  """

                if t == TEST_T:
                    print('---------2---------')
                    print(self.Transmission[t].sum(axis=0))
                    print(Netloadt)
                    print(self.SPower_nodal[t])
                    print(self.GFlexible_nodal[t])
                    """print(Discharget_max_nodal)
                    print(Charget_max_nodal) """
                    """ print('---------2.5---------')
                    print(SPowert_nodal)
                    print(Flexiblet_nodal) """

            #self.Deficit_nodal[t] = np.maximum(Netloadt - self.SPower_nodal[t] - self.GFlexible_nodal[t], 0) 
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

                if t == TEST_T:
                    print('---------2.5---------')
                    print(self.Transmission[t].sum(axis=0))
                    print(Netloadt)
                    print(self.SPower_nodal[t])
                    print(self.GFlexible_nodal[t])

            self.Spillage_nodal[t] = -1 * np.minimum(0, Netloadt - np.minimum(self.SPower_nodal[t], 0))

            """ if t == TEST_T:
                print('---------3---------')
                print(SPowert_nodal)
                print(Surplust) """

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

                if t == TEST_T:
                    print('---------4---------')
                    print(self.Transmission[t].sum(axis=0))
                    #print(Fillt)
                    print(Netloadt)
                    print(self.SPower_nodal[t])
                    print(self.GFlexible_nodal[t])
                    print('---------4.5---------')
                    """ print(SPowert_nodal)
                    print(Flexiblet_nodal)  """
            
            # Apportion to individual storages/flexible 
            #trans_sum = self.Transmission[t].sum(axis=0)
            self.Deficit_nodal[t] = np.maximum(Netloadt - self.SPower_nodal[t] - self.GFlexible_nodal[t], 0) ###### CHECK IF THIS DEFICIT CALC IS RIGHT
            
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
                               - np.maximum(self.SPower[t], 0) * self.storage_d_efficiencies * self.resolution 
                               - np.minimum(self.SPower[t], 0) * self.storage_c_efficiencies * self.resolution)
            
            if t == TEST_T:
                print('-------------5-------------')
                print(self.Deficit_nodal[t])
                print(self.SPower[t]) 
                print(self.GFlexible[t]) 
                print(self.SPower_nodal[t]) 
                print(self.GFlexible_nodal[t]) 

            if ((self.Storage[t] < 1e-3).any() and (self.Deficit_nodal[t].sum() > 1e-6)) or ((self.CPHS - self.Storage[t] < 1e-3).any() and (self.Spillage_nodal[t].sum() > 1e-6)):
                if precharge_energy < 1e-6:
                    t_pre_end = t
                    if (self.Deficit_nodal[t].sum() > 1e-6):
                        deficit_precharging = True
                    else:
                        deficit_precharging = False

                precharge_mask = precharge_mask | (self.Storage[t] < 1e-3) | (self.CPHS - self.Storage[t] < 1e-3)
                perform_precharge = False
                precharge_energy += self.Deficit_nodal[t].sum() + self.Spillage_nodal[t].sum()
            else:
                perform_precharge = True 

    def _check_feasibility(self, Netload):
        # Check max_surplus is greater than 0 and less than 50% total energy
        max_surplus = -1 * np.minimum(0,Netload).sum() + self.CFlexible.sum()*self.intervals - np.maximum(0,Netload).sum()
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