import numpy as np
from typing import List
import time

from firm_ce.network import get_transmission_flows_t
from firm_ce.constants import JIT_ENABLED, EPSILON_FLOAT64, NP_FLOAT_MAX
from firm_ce.components.costs import calculate_costs
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
        ('max_frequency', float64),
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
        self.max_frequency = max_frequency
        self.storage_durations = storage_durations
        self.storage_costs = storage_costs
        self.storage_d_efficiencies = storage_d_efficiencies
        self.storage_c_efficiencies = storage_c_efficiencies

        self.Storage = np.zeros((self.intervals,len(storage_ids)), dtype=np.float64)

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
        self.balancing_costs = np.hstack(
                                        (self.storage_costs, 
                                        self.generator_costs[:, flexible_mask])
                                        )

        

        """ nodal_durations = 
        for idx in range(len(storage_durations)):
            if storage_durations[idx] > 0:
                self.CPHS[idx] = self.CPHP[idx] * storage_durations[idx] """

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

        self.GFlexible_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Spillage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.NetBalancing_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
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

        # Frequency attributes
        self.balancing_W_cutoffs = np.zeros((self.nodes, self.balancing_W_x_nodal.shape[1] + 2), dtype=np.float64)

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
            for n, node_idx in enumerate(node_array):
                result[t, node_idx] += generation_array[t, n]

        return result
    
    def _fill_nodal_array_1d(self, capacity_array, node_array):
        result = np.zeros(self.nodes, dtype=np.float64)

        for n, node_idx in enumerate(node_array):
            result[node_idx] += capacity_array[n]

        return result

    def _reliability(self):
        Netload = (self.MLoad - self.GPV_nodal - self.GWind_nodal - self.GBaseload_nodal)

        Balancing = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Discharge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Charge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Storage = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Storage[-1] = 0.5*self.CPHS_nodal

        for t in range(self.intervals):  
            Netload_copperplate = Netload[t].sum()  
            if Netload_copperplate < 0:
                Charge[t] = np.minimum(
                                np.minimum(
                                    -1 * min(0, Netload_copperplate) * ((self.CPHS_nodal-Storage[t-1]) / (self.CPHS_nodal-Storage[t-1] + 1e-6).sum()), 
                                    (self.CPHS_nodal-Storage[t-1])/self.resolution
                                ), 
                                self.CPHP_nodal
                            )  
            else:
                Balancing[t] = np.minimum(
                                    np.minimum(
                                        max(0, Netload_copperplate) * (Storage[t-1] / (Storage[t-1].sum() + 1e-6)), 
                                        Storage[t-1]/self.resolution + self.CFlexible_nodal
                                    ), 
                                    self.CPHP_nodal + self.CFlexible_nodal
                                )
                
                Discharge[t] = np.minimum(
                                    np.minimum(
                                        max(0, Netload_copperplate) * (Storage[t-1] / (Storage[t-1].sum() + 1e-6)), 
                                        Storage[t-1]/self.resolution
                                    ), 
                                    self.CPHP_nodal
                                ) 

            Storage[t] = Storage[t-1] - (Discharge[t] - Charge[t]) * self.resolution
        
        np.savetxt("results/R_Netload.csv", Netload, delimiter=",")
        np.savetxt("results/R_NetBalancing_nodal.csv", Balancing - Charge, delimiter=",")
        np.savetxt("results/R_Balancing.csv", Balancing, delimiter=",")
        np.savetxt("results/R_Discharge.csv", Discharge, delimiter=",")
        np.savetxt("results/R_Charge.csv", Charge, delimiter=",")
        np.savetxt("results/R_Storage.csv", Storage, delimiter=",")
        
        return Netload, Balancing - Charge

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
    
    def _filter_balancing_profiles(self, NetBalancing_nodal):
        balancing_p_profiles_ifft = np.zeros((self.intervals,self.nodes,max(self.nodal_balancing_count)), dtype=np.float64)
        
        for node_idx in range(self.nodes):
            frequency_profile_p = frequency.get_frequency_profile(NetBalancing_nodal[:,node_idx])
            
            peak_mask, noise_mask = cwt.cwt_peak_detection(
                frequency.get_normalised_profile(frequency_profile_p)
            )
            #np.savetxt(f"results/peak_mask_{node_idx}.csv", peak_mask, delimiter=",")

            for balancing_i in range(self.nodal_balancing_count[node_idx]):
                if abs(self.balancing_W_cutoffs[node_idx, balancing_i] - self.max_frequency) <= EPSILON_FLOAT64:
                    break
                
                balancing_p_profiles_ifft[:,node_idx,balancing_i] = frequency.get_timeseries_profile(
                    frequency.get_filtered_frequency(
                        frequency_profile_p, 
                        peak_mask * frequency.get_bandpass_filter(
                                                        self.balancing_W_cutoffs[node_idx, balancing_i], 
                                                        self.balancing_W_cutoffs[node_idx, balancing_i+1], 
                                                        frequency.get_frequencies(
                                                                        self.intervals, 
                                                                        self.resolution
                                                                    )
                                                    )
                        )
                    )
                
            # Apportion dc offset and noise to long-duration
            balancing_p_profiles_ifft[:, node_idx, :] = frequency.apportion_nodal_noise(
                balancing_p_profiles_ifft[:, node_idx, :], 
                (frequency.get_timeseries_profile(frequency.get_dc_offset(frequency_profile_p)) 
                                  + frequency.get_timeseries_profile(frequency.get_filtered_frequency(
                                                                        frequency_profile_p, 
                                                                        noise_mask
                                                                    )
                                    )
                )
            )

            #np.savetxt(f"results/magnitudes_{node_idx}.csv", frequency.get_normalised_profile(frequency_profile_p), delimiter=",")

        return balancing_p_profiles_ifft
    
    def _determine_constrained_balancing(self, balancing_p_profiles_ifft):
        storage_p_profiles = np.zeros((self.intervals, len(self.storage_ids)), dtype=np.float64)
        flexible_p_profiles = np.zeros((self.intervals, len(self.flexible_ids)), dtype=np.float64)

        for node_idx in range(self.nodes):
            node_balancing_order = self.balancing_order[self.balancing_nodes == node_idx]

            node_balancing_permutation = frequency.order_balancing(
                node_balancing_order, 
                self.balancing_e_constraints[node_balancing_order],
                self.balancing_costs[3, node_balancing_order], # include fuel costs
                )

            balancing_p_profile_option = frequency.apply_balancing_constraints(
                balancing_p_profiles_ifft[:, node_idx, :len(node_balancing_order)], 
                self.balancing_e_constraints[node_balancing_permutation],
                self.balancing_d_efficiencies[node_balancing_permutation], 
                self.balancing_c_efficiencies[node_balancing_permutation], 
                self.balancing_d_constraint[node_balancing_permutation],
                self.balancing_c_constraint[node_balancing_permutation],
                self.resolution,
                )
            
            flexible_order_offset = len(self.storage_ids) + 1
            for i in range(len(node_balancing_order)):
                if node_balancing_permutation[i] < flexible_order_offset - 1:
                    storage_p_profiles[:, node_balancing_permutation[i]] = balancing_p_profile_option[:, i]
                else:
                    flexible_p_profiles[:, node_balancing_permutation[i] - flexible_order_offset] = balancing_p_profile_option[:, i]
        
        np.savetxt(f"results/storage_p_profiles.csv", storage_p_profiles, delimiter=",")
        return storage_p_profiles, flexible_p_profiles
    
    def _transmission_balancing(self, Netload, storage_p_profiles, flexible_p_profiles):
        flexible_p_profiles = np.full(flexible_p_profiles.shape, 0.5, dtype=np.float64) ####### DEBUG
        TEST_T = 9235 ####### DEBUG

        network = self.network

        NetStoragePower_nodal = self._fill_nodal_array_2d(storage_p_profiles, self.storage_nodes)
        Flexible_nodal = self._fill_nodal_array_2d(flexible_p_profiles, self.flexible_nodes)
        
        Netload = Netload - NetStoragePower_nodal - Flexible_nodal
        shape2d = intervals, nodes = Netload.shape

        Scapacity = self.CPHS
        Fcapacity = self.CFlexible 
        Pcapacity = self.CPHP
        Fcapacity_nodal = self._fill_nodal_array_1d(Fcapacity, self.flexible_nodes)

        storage_order = self.balancing_order[self.balancing_storage_tag]
        flexible_order = self.balancing_order[self.balancing_flexible_tag]
        F_variable_costs = self.balancing_costs[3,flexible_order] ##### ADD FUEL COSTS

        Hcapacity = self.CTrans
        ntrans = len(self.CTrans)

        Flexible = np.zeros_like(flexible_p_profiles, dtype=np.float64)
        SPower = np.zeros_like(storage_p_profiles, dtype=np.float64)
        Storage = np.zeros_like(storage_p_profiles, dtype=np.float64)       
        Transmission = np.zeros((intervals, ntrans, nodes), dtype = np.float64)

        Storaget_1 = 0.5*Scapacity

        for t in range(intervals):
            Storaget_p_lb = Storaget_1 / (self.resolution * self.storage_d_efficiencies)
            Storaget_p_ub = - (Scapacity - Storaget_1) / (self.resolution * self.storage_c_efficiencies)
            Discharget_max = np.minimum(Pcapacity, Storaget_p_lb)
            Discharget_max_nodal = self._fill_nodal_array_1d(Discharget_max, self.storage_nodes)
            Charget_max = np.minimum(Pcapacity, -Storaget_p_ub)
            Charget_max_nodal = self._fill_nodal_array_1d(Charget_max, self.storage_nodes)

            SPowert = storage_p_profiles[t, :].copy()
            Flexiblet = flexible_p_profiles[t, :].copy()
            SPowert_nodal = self._fill_nodal_array_1d(SPowert, self.storage_nodes)
            Flexiblet_nodal = self._fill_nodal_array_1d(Flexiblet, self.flexible_nodes)

            SPowert_update = np.maximum(np.minimum(storage_p_profiles[t, :], Storaget_p_lb), Storaget_p_ub) - SPowert
            Flexiblet_update = np.maximum(flexible_p_profiles[t, :], 0) - Flexiblet
            SPowert_update_nodal = self._fill_nodal_array_1d(SPowert_update, self.storage_nodes)
            Flexiblet_update_nodal = self._fill_nodal_array_1d(Flexiblet_update, self.flexible_nodes)

            Netloadt = Netload[t].copy()

            Deficitt = np.maximum(Netloadt - SPowert_update_nodal - Flexiblet_update_nodal, 0)
            Transmissiont=np.zeros((ntrans, nodes), dtype=np.float64)

            if t == TEST_T:
                print('---------1---------')
                print(Netloadt)
                print(SPowert_update_nodal)
                print(Flexiblet_update_nodal)
                print(Deficitt)
            
            if Deficitt.sum() > 1e-6:
                # Fill deficits with transmission allowing drawing down from neighbours storage reserves                
                Surplust = (
                    -1 * np.minimum(0, Netloadt)
                    + (Discharget_max_nodal - (SPowert_nodal + SPowert_update_nodal))
                    + (Fcapacity_nodal - (Flexiblet_nodal + Flexiblet_update_nodal))
                )

                Transmissiont = get_transmission_flows_t(
                    Deficitt, Surplust, Hcapacity, network, self.networksteps,
                    np.maximum(0, Transmissiont), np.minimum(0, Transmissiont)
                )

                Netloadt -= Transmissiont.sum(axis=0)

                SPowert_update_nodal = np.maximum(
                    np.minimum(Netloadt, Discharget_max_nodal - SPowert_nodal),
                    -Charget_max_nodal - SPowert_nodal
                )

                Flexiblet_update_nodal = np.minimum(
                    np.maximum(Netloadt - SPowert_update_nodal, 0),
                    Fcapacity_nodal - Flexiblet_nodal
                )

                if t == TEST_T:
                    print('---------2---------')
                    print(Netloadt)
                    print(SPowert_update_nodal)
                    print(Flexiblet_update_nodal)
                    print('---------2.5---------')
                    print(SPowert_nodal)
                    print(Flexiblet_nodal)

            #SPowert_nodal += SPowert_update_nodal
            Surplust = -1 * np.minimum(0, Netloadt - np.minimum(SPowert_update_nodal, 0))

            if t == TEST_T:
                print('---------3---------')
                print(SPowert_nodal)
                print(Surplust)

            if Surplust.sum() > 1e-6:
                Fillt = SPowert_update_nodal + (Charget_max_nodal + SPowert_nodal + SPowert_update_nodal)

                Transmissiont = get_transmission_flows_t(
                    Fillt, Surplust, Hcapacity, network, self.networksteps,
                    np.maximum(0, Transmissiont), np.minimum(0, Transmissiont)
                )

                Netloadt = Netload[t] - Transmissiont.sum(axis=0)

                SPowert_update_nodal = np.maximum(
                    np.minimum(Netloadt, Discharget_max_nodal - SPowert_nodal),
                    -Charget_max_nodal - SPowert_nodal
                )

                Flexiblet_update_nodal = np.minimum(
                    np.maximum(Netloadt - SPowert_update_nodal, 0),
                    Fcapacity_nodal - Flexiblet_nodal
                )   

                if t == TEST_T:
                    print('---------4---------')
                    print(Fillt)
                    print(Netloadt)
                    print(SPowert_update_nodal)
                    print(Flexiblet_update_nodal)
                    print('---------4.5---------')
                    print(SPowert_nodal)
                    print(Flexiblet_nodal) 
            
            # Apportion to individual storages/flexible 
            trans_sum = Transmissiont.sum(axis=0)
            for node in range(self.nodes):
                # Apportion storage
                storage_mask = self.storage_nodes == node
                if np.any(storage_mask):
                    Storaget_1_node = Storaget_1[storage_mask]
                    storage_order_node = storage_order[storage_mask]                    
                    sorted_indices = np.argsort(Storaget_1_node)
                    
                    L = sorted_indices.shape[0]
                    order_indices = np.empty(2 * L, dtype=sorted_indices.dtype)
                    for i in range(L):
                        order_indices[i] = L - sorted_indices[i] - 1
                    for i in range(L):
                        order_indices[L + i] = sorted_indices[i]

                    # If there is a deficit at this node, remove intra‐node charging.
                    if Netload[t, node] - trans_sum[node] - SPowert_update_nodal[node] - Flexiblet_update_nodal[node] > 0:
                        spower_node = SPowert[storage_mask]
                        spower_new = np.maximum(spower_node, 0)
                        change = spower_new.sum() - spower_node.sum()
                        Flexiblet_update_nodal[node] = max(0, Flexiblet_update_nodal[node] - change)
                        SPowert[storage_mask] = spower_new

                    for idx in order_indices:
                        storage_order_i = storage_order_node[idx]
                        current = SPowert[storage_order_i]
                        
                        lower_bound = max(Storaget_p_ub[storage_order_i], -Pcapacity[storage_order_i])
                        upper_bound = min(Storaget_p_lb[storage_order_i], Pcapacity[storage_order_i])
                        new_value = helpers.scalar_clamp(current + SPowert_update_nodal[node], lower_bound, upper_bound)
                        
                        SPowert_update_nodal[node] -= (new_value - current)
                        SPowert[storage_order_i] = new_value

                # Apportion flexible
                flexible_mask = self.flexible_nodes == node
                if np.any(flexible_mask):
                    flexible_variable_costs_node = F_variable_costs[flexible_mask]
                    flexible_order_node = flexible_order[flexible_mask]
                    sorted_indices = np.argsort(flexible_variable_costs_node)

                    L = sorted_indices.shape[0]
                    order_indices = np.empty(2 * L, dtype=sorted_indices.dtype)
                    for i in range(L):
                        order_indices[i] = L - sorted_indices[i] - 1
                    for i in range(L):
                        order_indices[L + i] = sorted_indices[i]
                    
                    for idx in order_indices:                        
                        flexible_order_i = flexible_order_node[idx] - len(storage_order)
                        current = Flexiblet[flexible_order_i]
                        new_value = helpers.scalar_clamp(current + Flexiblet_update_nodal[node], 0, Fcapacity[flexible_order_i])
                        Flexiblet_update_nodal[node] -= (new_value - current)
                        Flexiblet[flexible_order_i] = new_value

            Storaget = Storaget_1 - np.maximum(SPowert, 0) * self.storage_d_efficiencies * self.resolution \
                   - np.minimum(SPowert, 0) * self.storage_c_efficiencies * self.resolution
            
            Storaget_1 = Storaget.copy()

            Flexible[t] = Flexiblet
            SPower[t] = SPowert 
            Transmission[t] = Transmissiont
            Storage[t] = Storaget
        
        SPower_updated_nodal = self._fill_nodal_array_2d(SPower, self.storage_nodes)
        Flexible_updated_nodal = self._fill_nodal_array_2d(Flexible, self.flexible_nodes)
        ImpExp = Transmission.sum(axis=1)    
        Deficit = np.maximum(0, Netload - ImpExp - (Flexible_updated_nodal - Flexible_nodal) - (SPower_updated_nodal - NetStoragePower_nodal))
        Spillage = -1 * np.minimum(0, Netload - ImpExp - (SPower_updated_nodal - NetStoragePower_nodal))   

        self.Spillage_nodal = Spillage        
        self.Deficit_nodal = Deficit
        self.Import_nodal = np.maximum(0, ImpExp)
        self.Export_nodal = -1 * np.minimum(0, ImpExp)

        self.TFlows = (Transmission).sum(axis=2)
        self.GDischarge = np.maximum(SPower, 0)
        self.GFlexible = Flexible
        self.Storage = Storage 

        np.savetxt("results/Netload.csv", Netload, delimiter=",")
        np.savetxt("results/ImpExp.csv", ImpExp, delimiter=",")
        np.savetxt("results/Deficit.csv", self.Deficit_nodal, delimiter=",")
        np.savetxt("results/Spillage.csv", self.Spillage_nodal, delimiter=",")
        np.savetxt("results/Storage.csv", self.Storage, delimiter=",")
        np.savetxt("results/SPower_update.csv", SPower - storage_p_profiles, delimiter=",")
        np.savetxt("results/Flexible_update.csv", Flexible - flexible_p_profiles, delimiter=",")
        np.savetxt("results/SPower.csv", SPower, delimiter=",")
        np.savetxt("results/Flexible.csv", Flexible, delimiter=",")
        #OUTPUT = np.vstack((Netload[:18000,0],ImpExp[:18000,0],SPower[:18000,0] - storage_p_profiles[:18000,0],SPower[:18000,5] - storage_p_profiles[:18000,5],Flexible[:18000,0] - flexible_p_profiles[:18000,0],self.Deficit_nodal[:18000,0],self.Spillage_nodal[:18000,0],self.Storage[:18000,0],self.Storage[:18000,5],SPower[:18000,0],SPower[:18000,5],Flexible[:18000,0]))
        OUTPUT = np.concatenate((self.MLoad[:18000,:],self.GPV[:18000,:],self.GWind[:18000,:],self.GBaseload[:18000,:],Flexible[:18000,:],SPower[:18000,:],self.Deficit_nodal[:18000,:],self.Spillage_nodal[:18000,:],self.Storage[:18000,:],ImpExp[:18000,:]), axis=1)
        OUTPUT2 = np.concatenate((Netload[:18000,:],ImpExp[:18000,:],(Flexible - flexible_p_profiles)[:18000,:],(SPower - storage_p_profiles)[:18000,:],self.Deficit_nodal[:18000,:],self.Spillage_nodal[:18000,:],Flexible[:18000,:],SPower[:18000,:],self.Storage[:18000,:]), axis=1)
        
        np.savetxt("results/OUTPUT.csv", 1000*OUTPUT, delimiter=",")
        np.savetxt("results/OUTPUT2.csv", 1000*OUTPUT2, delimiter=",")

        return self.Deficit_nodal, np.abs(self.TFlows)

    def _objective(self) -> List[float]:
        Netload, NetBalancing_nodal = self._reliability()

        balancing_p_profiles_ifft = self._filter_balancing_profiles(NetBalancing_nodal)
        storage_p_profiles, flexible_p_profiles = self._determine_constrained_balancing(balancing_p_profiles_ifft)

        deficit, TFlowsAbs = self._transmission_balancing(Netload, storage_p_profiles, flexible_p_profiles)
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution / self.years - self.allowance) * 1000000

        self._calculate_annual_generation()
        cost = self._calculate_costs()

        loss = TFlowsAbs.sum(axis=0) * self.TLoss
        loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - loss) / 1000 # $/MWh
        
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
                    generator_costs,
                    storage_ids,
                    storage_nodes,
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
                                generator_costs,
                                storage_ids,
                                storage_nodes,
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
                generator_costs,
                storage_ids,
                storage_nodes,
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
                                generator_costs,
                                storage_ids,
                                storage_nodes,
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