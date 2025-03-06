import numpy as np
from typing import List
import time

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
        ('balancing_storage_tag', boolean[:]),
        ('balancing_flexible_tag', boolean[:]),
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

        self.MLoad = MLoad/1000.
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
        self.solar_nodes = solar_nodes
        self.wind_nodes = wind_nodes
        self.flexible_nodes = flexible_nodes 
        self.baseload_nodes = baseload_nodes
        self.storage_nodes = self.storage_nodes

        self.CFlexible_nodal = self._fill_nodal_array_1d(self.CFlexible, self.flexible_nodes)
        self.CPHP_nodal = self._fill_nodal_array_1d(self.CPHP, self.storage_nodes)
        self.CPHS_nodal = self._fill_nodal_array_1d(self.CPHS, self.storage_nodes)    
        self.GPV = self.CPV[np.newaxis, :] * TSPV
        self.GPV_nodal = self._fill_nodal_array_2d(self.GPV, self.solar_nodes)
        self.GWind = self.CWind[np.newaxis, :] * TSWind 
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
        self.balancing_p_profiles_ifft = np.zeros((self.intervals,self.nodes,max(self.nodal_balancing_count)), dtype=np.float64)
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
        Netload = (self.MLoad - self.GPV - self.GWind - self.GBaseload_nodal)

        efficiency = 1.0 # Assume no efficiency losses for the initial rough storage

        Balancing = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Discharge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Charge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Storage = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        Storage[-1] = 0.5*self.CPHS_nodal

        for t in range(self.intervals):
            Balancing[t] = np.minimum(np.minimum(np.maximum(0, Netload[t]), Storage[t-1]/self.resolution + self.CFlexible_nodal), self.CPHP_nodal + self.CFlexible_nodal)
            Discharge[t] = np.minimum(np.minimum(np.maximum(0, Netload[t]), Storage[t-1]/self.resolution), self.CPHP_nodal) 
            Charge[t] = np.minimum(np.minimum(-1 * np.minimum(0, Netload[t]), (self.CPHS_nodal-Storage[t-1])/efficiency/self.resolution), self.CPHP_nodal)
            Storage[t] = Storage[t-1] - Discharge[t] * self.resolution + Charge[t] * self.resolution * efficiency
        
        """ np.savetxt("results/R_Netload.csv", Netload, delimiter=",")
        np.savetxt("results/R_NetBalancing_nodal.csv", NetBalancing_nodal, delimiter=",")
        np.savetxt("results/R_Balancing.csv", Balancing, delimiter=",")
        np.savetxt("results/R_Discharge.csv", Discharge, delimiter=",")
        np.savetxt("results/R_Charge.csv", Charge, delimiter=",")
        np.savetxt("results/R_Storage.csv", Storage, delimiter=",") """
        
        return Netload, Balancing - Charge #NetBalancing_nodal=Balancing-Charge

    def _calculate_costs(self):
        solution_cost = calculate_costs(self)
        return solution_cost
    
    def _apportion_nodal_array(self, capacity_array, nodal_generation, node_array):
        
        cap_node_sum = np.zeros(self.nodes, dtype=np.float64)
        for i in range(len(node_array)):
            cap_node_sum[node_array[i]] += capacity_array[i]
        
        result = np.zeros((len(node_array), self.intervals), dtype=np.float64)
        for i in range(len(node_array)):
            if cap_node_sum[node_array[i]] > 0.0:
                result[i] = capacity_array[i] / cap_node_sum[node_array[i]] * nodal_generation[:, node_array[i]]
            else:
                # If the node's total capacity is zero, fall back to "no scaling"
                result[i] = nodal_generation[:, node_array[i]]

        return result.T
    
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
    
    def _filter_balancing_profiles(self, NetBalancing_nodal):
        balancing_p_profiles_ifft = np.zeros((self.intervals,self.nodes,max(self.nodal_balancing_count)), dtype=np.float64)
        
        for node_idx in range(self.nodes):
            frequency_profile_p = frequency.get_frequency_profile(NetBalancing_nodal[:,node_idx])
            
            peak_mask, noise_mask = cwt.cwt_peak_detection(frequency.get_normalised_profile(frequency_profile_p))

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
                frequency.get_timeseries_profile(frequency.get_dc_offset(frequency_profile_p)) + 
                frequency.get_timeseries_profile(frequency.get_filtered_frequency(frequency_profile_p, noise_mask)))

        return balancing_p_profiles_ifft
    
    def _determine_constrained_balancing(self, balancing_p_profiles_ifft):
        storage_p_profiles = np.zeros((self.intervals, len(self.storage_ids)), dtype=np.float64)
        flexible_p_profiles = np.zeros((self.intervals, len(self.flexible_ids)), dtype=np.float64)

        for node_idx in range(self.nodes):
            node_balancing_order = self.balancing_order[self.balancing_nodes == node_idx]

            node_balancing_permutation = frequency.order_balancing(
                node_balancing_order, 
                self.balancing_e_constraints[node_balancing_order],
                self.balancing_costs[3, node_balancing_order], # includes fuel costs
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
            
            flexible_order_offset = len(self.storage_ids) 
            for i in range(len(node_balancing_order)):
                if node_balancing_permutation[i] < flexible_order_offset - 1:
                    storage_p_profiles[:, node_balancing_permutation[i]] = balancing_p_profile_option[:, i]
                else:
                    flexible_p_profiles[:, node_balancing_permutation[i] - flexible_order_offset] = balancing_p_profile_option[:, i]

        return storage_p_profiles, flexible_p_profiles
    
    def _transmission_balancing(self, Netload, storage_p_profiles, flexible_p_profiles):
        NetStoragePower_nodal = self._fill_nodal_array_2d(storage_p_profiles, self.storage_nodes)
        Flexible_nodal = self._fill_nodal_array_2d(flexible_p_profiles, self.flexible_nodes)
        
        Netload = Netload - NetStoragePower_nodal - Flexible_nodal

        Fcapacity_nodal = self._fill_nodal_array_1d(self.CFlexible, self.flexible_nodes)
        F_variable_costs = self.balancing_costs[3,self.balancing_order[self.balancing_flexible_tag]] ##### ADD FUEL COSTS

        Flexible = np.zeros_like(flexible_p_profiles, dtype=np.float64)
        SPower = np.zeros_like(storage_p_profiles, dtype=np.float64)
        self.Storage_nodal = np.zeros_like(storage_p_profiles, dtype=np.float64)       
        Transmission = np.zeros((self.intervals, self.lines, self.nodes), dtype = np.float64)
        
        self.Storage_nodal[-1] = 0.5*self.CPHS
        
        self.Deficit_nodal = np.zeros_like(Netload)
        self.Spillage_nodal = np.zeros_like(Netload)
        
        for t in range(self.intervals):
            Storaget_p_lb = self.Storage_nodal[t-1] / (self.resolution * self.storage_d_efficiencies)
            Storaget_p_ub = - (self.CPHS - self.Storage_nodal[t-1]) / (self.resolution * self.storage_c_efficiencies)
            Discharget_max_nodal = self._fill_nodal_array_1d(np.minimum(self.CPHP, Storaget_p_lb), self.storage_nodes)
            Charget_max_nodal = self._fill_nodal_array_1d(np.minimum(self.CPHP, -Storaget_p_ub), self.storage_nodes)

            SPower[t] = storage_p_profiles[t, :].copy()
            Flexible[t] = flexible_p_profiles[t, :].copy()
            SPowert_nodal = self._fill_nodal_array_1d(SPower[t], self.storage_nodes)
            Flexiblet_nodal = self._fill_nodal_array_1d(Flexible[t], self.flexible_nodes)

            SPowert_update_nodal = self._fill_nodal_array_1d(np.maximum(np.minimum(storage_p_profiles[t, :], Storaget_p_lb), Storaget_p_ub) - SPower[t], self.storage_nodes)
            Flexiblet_update_nodal = self._fill_nodal_array_1d(np.maximum(flexible_p_profiles[t, :], 0) - Flexible[t], self.flexible_nodes)

            # This one is actually necessary
            Netloadt = Netload[t].copy()

            # Avoiding extra memory allocation. This will be overwritten after loop
            self.Deficit_nodal[t] = np.maximum(Netloadt - SPowert_update_nodal - Flexiblet_update_nodal, 0)
            
            if self.Deficit_nodal[t].sum() > 1e-6:
                # Fill deficits with transmission allowing drawing down from neighbours storage reserves                
                self.Spillage_nodal[t] = (
                    -np.minimum(0, Netloadt)
                    + (Discharget_max_nodal - SPowert_nodal + SPowert_update_nodal)
                    + (Fcapacity_nodal - Flexiblet_nodal + Flexiblet_update_nodal)
                )

                Transmission[t] = get_transmission_flows_t(
                    self.Deficit_nodal[t], self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, Transmission[t]), np.minimum(0, Transmission[t])
                )

                Netloadt -= Transmission[t].sum(axis=0)

                SPowert_update_nodal = np.maximum(
                    np.minimum(Netloadt, Discharget_max_nodal - SPowert_nodal),
                    -Charget_max_nodal - SPowert_nodal
                )

                Flexiblet_update_nodal = np.minimum(
                    np.maximum(Netloadt - SPowert_update_nodal, 0),
                    Fcapacity_nodal - Flexiblet_nodal
                )

            SPowert_nodal += SPowert_update_nodal
            self.Spillage_nodal[t] = -np.minimum(0, Netloadt - np.minimum(SPowert_update_nodal, 0))

            if self.Spillage_nodal[t].sum() > 1e-6:
                Transmission[t] = get_transmission_flows_t(
                    SPowert_update_nodal + (Charget_max_nodal + SPowert_nodal),
                    self.Spillage_nodal[t], self.CTrans, self.network, self.networksteps,
                    np.maximum(0, Transmission[t]), np.minimum(0, Transmission[t])
                )

                Netloadt = Netload[t] - Transmission[t].sum(axis=0)

                SPowert_update_nodal = np.maximum(
                    np.minimum(Netloadt, Discharget_max_nodal - SPowert_nodal),
                    -Charget_max_nodal - SPowert_nodal
                )

                Flexiblet_update_nodal = np.minimum(
                    np.maximum(Netloadt - SPowert_update_nodal, 0),
                    Fcapacity_nodal - Flexiblet_nodal
                )    
            
            # Apportion to individual storages/flexible 
            for node in range(self.nodes):
                # Apportion storage
                storage_mask = self.storage_nodes == node
                if np.any(storage_mask):
                    sorted_indices = np.argsort(self.Storage_nodal[t-1][storage_mask])
                    
                    L = sorted_indices.shape[0]
                    order_indices = np.empty(2 * L, dtype=sorted_indices.dtype)
                    for i in range(L):
                        order_indices[i] = L - sorted_indices[i] - 1
                    for i in range(L):
                        order_indices[L + i] = sorted_indices[i]

                    # If there is a deficit at this node, remove intra‐node charging.
                    if Netload[t, node] - Transmission[t].sum(axis=0)[node] - SPowert_update_nodal[node] - Flexiblet_update_nodal[node] > 0:
                        Flexiblet_update_nodal[node] = max(0, 
                            Flexiblet_update_nodal[node] - np.maximum(SPower[t][storage_mask], 0).sum() - SPower[t][storage_mask].sum())
                        SPower[t][storage_mask] = np.maximum(SPower[t][storage_mask], 0)

                    for idx in order_indices:
                        # I think this is a useful dummy variable to keep because of how expensive the indexing is
                        storage_order_i = self.balancing_order[self.balancing_storage_tag][storage_mask][idx]
                        
                        new_value = helpers.scalar_clamp(
                            SPower[t][storage_order_i] + SPowert_update_nodal[node], 
                            max(Storaget_p_ub[storage_order_i], -self.CPHP[storage_order_i]), 
                            min(Storaget_p_lb[storage_order_i], self.CPHP[storage_order_i])
                        )
                        
                        SPowert_update_nodal[node] -= (new_value - SPower[t][storage_order_i])
                        SPower[t][storage_order_i] = new_value

                # Apportion flexible
                flexible_mask = self.flexible_nodes == node
                if np.any(flexible_mask):
                    sorted_indices = np.argsort(F_variable_costs[flexible_mask])

                    L = sorted_indices.shape[0]
                    order_indices = np.empty(2 * L, dtype=sorted_indices.dtype)
                    for i in range(L):
                        order_indices[i] = L - sorted_indices[i] - 1
                    for i in range(L):
                        order_indices[L + i] = sorted_indices[i]
                    
                    for idx in order_indices:                        
                        flexible_order_i = self.balancing_order[self.balancing_flexible_tag][flexible_mask][idx] - len(self.balancing_order[self.balancing_storage_tag])
                        new_value = helpers.scalar_clamp(
                            Flexible[t][flexible_order_i] + Flexiblet_update_nodal[node], 
                            0, 
                            self.CFlexible[flexible_order_i]
                        )
                        Flexiblet_update_nodal[node] -= (new_value - Flexible[t][flexible_order_i])
                        Flexible[t][flexible_order_i] = new_value

            self.Storage_nodal[t] = (self.Storage_nodal[t-1] 
                                     - np.maximum(SPower[t], 0) * self.storage_d_efficiencies * self.resolution 
                                     - np.minimum(SPower[t], 0) * self.storage_c_efficiencies * self.resolution)

        SPower_updated_nodal = self._fill_nodal_array_2d(SPower, self.storage_nodes)
        ImpExp = Transmission.sum(axis=1)    
        self.Deficit_nodal = np.maximum(0, Netload - ImpExp - (self._fill_nodal_array_2d(Flexible, self.flexible_nodes) - Flexible_nodal) - (SPower_updated_nodal - NetStoragePower_nodal))
        self.Spillage_nodal = -1 * np.minimum(0, Netload - ImpExp - (SPower_updated_nodal - NetStoragePower_nodal))    

        self.Import_nodal = np.maximum(0, ImpExp)
        self.Export_nodal = -1 * np.minimum(0, ImpExp)
        self.TFlows = Transmission.sum(axis=2)

        """ np.savetxt("results/Netload.csv", Netload, delimiter=",")
        np.savetxt("results/ImpExp.csv", ImpExp, delimiter=",")
        np.savetxt("results/Deficit.csv", Deficit, delimiter=",")
        np.savetxt("results/Spillage.csv", Spillage, delimiter=",")
        np.savetxt("results/Storage.csv", Storage, delimiter=",")
        np.savetxt("results/SPower_update.csv", SPower - storage_p_profiles, delimiter=",")
        np.savetxt("results/Flexible_update.csv", Flexible - flexible_p_profiles, delimiter=",")
        np.savetxt("results/SPower.csv", SPower, delimiter=",")
        np.savetxt("results/Flexible.csv", Flexible, delimiter=",")
        OUTPUT = np.vstack((Netload[:18000,0],ImpExp[:18000,0],SPower[:18000,0] - storage_p_profiles[:18000,0],SPower[:18000,5] - storage_p_profiles[:18000,5],Flexible[:18000,0] - flexible_p_profiles[:18000,0],Deficit[:18000,0],Spillage[:18000,0],Storage[:18000,0],Storage[:18000,5],SPower[:18000,0],SPower[:18000,5],Flexible[:18000,0]))
        np.savetxt("results/OUTPUT.csv", OUTPUT.T, delimiter=",") """

        return self.Deficit_nodal, np.abs(self.TFlows) #TFlowsAbs = np.abs(self.TFlows)

    def _objective(self) -> List[float]:
        start_objective = time.time()
        Netload, NetBalancing_nodal = self._reliability()
        reliability_time = time.time()
        print(f"Reliability time: {reliability_time - start_objective:.4f} seconds")

        balancing_p_profiles_ifft = self._filter_balancing_profiles(NetBalancing_nodal)
        frequency_time = time.time()
        print(f"Frequency time: {frequency_time - reliability_time:.4f} seconds")
        storage_p_profiles, flexible_p_profiles = self._determine_constrained_balancing(balancing_p_profiles_ifft)
        constraints_time = time.time()
        print(f"Constraints time: {constraints_time - frequency_time:.4f} seconds")

        deficit, TFlowsAbs = self._transmission_balancing(Netload, storage_p_profiles, flexible_p_profiles)
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution - self.allowance) * 1000
        transmission_time = time.time()
        print(f"Transmission time: {transmission_time - constraints_time:.4f} seconds")

        self._apportion_nodal_generation()
        self._calculate_annual_generation()
        cost = self._calculate_costs()
        costs_time = time.time()
        print(f"Costs time: {costs_time - transmission_time:.4f} seconds")

        loss = TFlowsAbs.sum(axis=0) * self.TLoss
        loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - loss)
        end_objective = time.time()
        print(f"Objective solve time: {end_objective - start_objective:.4f} seconds")
        print("LCOE: ", lcoe, pen_deficit)

        raise KeyboardInterrupt
        return lcoe, pen_deficit

    def evaluate(self):
        self.lcoe, self.penalties = self._objective()
        self.evaluated=True 
        return self

@njit(parallel=False)
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