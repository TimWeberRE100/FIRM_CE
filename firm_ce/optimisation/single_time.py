import numpy as np
from typing import List

from firm_ce.network import get_transmission_flows_t
from firm_ce.constants import JIT_ENABLED, EPSILON_FLOAT64, NP_FLOAT_MAX
from firm_ce.components.costs import calculate_costs, annualisation_component
import firm_ce.network.frequency as frequency

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
        ('nodes_with_storage', int64[:]),
        ('max_frequency', float64),
        ('storage_durations', float64[:]),
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
        ('storage_e_W_x', float64[:]),

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
        ('storage_cost_ids', int64[:]),
        ('line_cost_ids', int64[:]),

        ('storage_p_profiles', float64[:,:,:]),
        ('storage_e_profiles', float64[:,:,:]),
        ('storage_e_W_x_nodal', float64[:,:]),
        ('storage_e_W_cutoffs', float64[:,:]),
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
                nodes_with_storage,
                max_frequency,
                storage_durations,
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
                storage_e_W_idx,
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
        self.nodes_with_storage = nodes_with_storage
        self.max_frequency = max_frequency
        self.storage_durations = storage_durations
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
        self.CPHS_nodal = x[storage_p_idx : storage_e_idx]
        self.CTrans = x[storage_e_idx : lines_idx]
        self.storage_e_W_x = x[lines_idx : ]

        print(self.CPV,self.CWind,self.CPHP,self.CPHS_nodal,self.CTrans,self.storage_e_W_x)

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

        self.CPHP_nodal = self._fill_nodal_array_1d(self.CPHP, self.storage_nodes)
        """ self.CPHS_nodal = self._fill_nodal_array_1d(self.CPHS, self.storage_nodes) """   
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
        self.Storage_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Deficit_nodal = np.zeros((self.intervals,self.nodes), dtype=np.float64)
        self.Import_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64) 
        self.Export_nodal = np.zeros((self.intervals,self.lines), dtype=np.float64)   

        self.nodal_storage_count = np.zeros(self.nodes, dtype=np.int64)
        for node_idx in range(len(self.nodal_storage_count)):
            for i in self.storage_nodes:
                if i == node_idx:
                    self.nodal_storage_count[node_idx]+=1

        self.storage_e_W_x_nodal = -1 * np.ones((self.nodes, max(self.nodal_storage_count)), dtype=np.float64)

        for node_idx in range(len(self.nodal_storage_count)):
            for current_node_count in range(self.nodal_storage_count[node_idx]-1):
                W_idx = sum(self.nodal_storage_count[:node_idx]) - len(self.nodal_storage_count[:node_idx]) + current_node_count if node_idx > 0 else current_node_count
                self.storage_e_W_x_nodal[node_idx, current_node_count] = self.storage_e_W_x[W_idx]

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
        self.storage_e_profiles_ifft = np.zeros((self.intervals,self.nodes,max(self.nodal_storage_count)), dtype=np.float64)
        self.storage_e_capacities_ifft = np.zeros((self.nodes,max(self.nodal_storage_count)), dtype=np.float64)
        self.storage_e_W_cutoffs = np.zeros((self.nodes, self.storage_e_W_x_nodal.shape[1] + 2), dtype=np.float64)  

        self.storage_p_profiles = np.zeros((self.intervals,len(self.CPHP)), dtype=np.float64)
        self.storage_e_profiles = np.zeros((self.intervals,len(self.CPHP)), dtype=np.float64)                
        self.storage_e_capacities = np.zeros(len(self.CPHP), dtype=np.float64)
        self.storage_p_capacities = self.CPHP

        #print(x)      

        for node_idx in self.storage_nodes:
            for W_idx in range(self.storage_e_W_x_nodal.shape[1]):
                if self.storage_e_W_x_nodal[node_idx,W_idx] > -0.5:
                    self.storage_e_W_cutoffs[node_idx,W_idx+1] = self.storage_e_W_x_nodal[node_idx,W_idx]
                else:
                    self.storage_e_W_cutoffs[node_idx,W_idx+1] = max_frequency
            self.storage_e_W_cutoffs[node_idx,-1] = max_frequency        

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
                Transmissiont = get_transmission_flows_t(Fillt, Surplust, Hcapacity, network, self.networksteps,
                                    np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))

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
    
    def _get_storage_capacities(self):
        self.storage_e_profiles_ifft = np.zeros((self.intervals,self.nodes,max(self.nodal_storage_count)), dtype=np.float64)
        self.storage_e_capacities_ifft = np.zeros((self.nodes,max(self.nodal_storage_count)), dtype=np.float64)
        
        """ np.savetxt("results/Charge_nodal.csv", self.Charge_nodal, delimiter=",")
        np.savetxt("results/Discharge_nodal.csv", self.Discharge_nodal, delimiter=",")
        np.savetxt("results/Storage_nodal.csv", self.Storage_nodal, delimiter=",") """
        
        for node_idx in range(self.nodes):
            frequency_profile_e = frequency.get_frequency_profile(self.Storage_nodal[:,node_idx])
            normalised_magnitudes_e = frequency.get_normalised_profile(frequency_profile_e)
            total_magnitude_e = sum(normalised_magnitudes_e)

            frequencies = frequency.get_frequencies(self.intervals, self.resolution)

            dc_offset_e = frequency.get_dc_offset(frequency_profile_e)
            dc_offset_e_timeseries = frequency.get_timeseries_profile(dc_offset_e)

            for storage_i in range(self.nodal_storage_count[node_idx]):
                if abs(self.storage_e_W_cutoffs[node_idx, storage_i] - self.max_frequency) <= EPSILON_FLOAT64:
                    break

                # Energy Capacity
                bandpass_filter_e = frequency.get_bandpass_filter(self.storage_e_W_cutoffs[node_idx, storage_i],self.storage_e_W_cutoffs[node_idx, storage_i+1], frequencies)
                filtered_magnitudes_e = frequency.get_filtered_frequency(normalised_magnitudes_e, bandpass_filter_e)
                unit_magnitude_e = sum(filtered_magnitudes_e)                
                self.storage_e_capacities_ifft[node_idx,storage_i] = self.CPHS_nodal[node_idx] * unit_magnitude_e / total_magnitude_e
                
                filtered_frequency_profile_e = frequency.get_filtered_frequency(frequency_profile_e, bandpass_filter_e)
                unit_timeseries_e = frequency.get_timeseries_profile(filtered_frequency_profile_e)
                self.storage_e_profiles_ifft[:,node_idx,storage_i] = unit_timeseries_e
                
            # Apportion dc offset
            self.storage_e_profiles_ifft[:,node_idx,0] = self.storage_e_profiles_ifft[:,node_idx,0] + dc_offset_e_timeseries

            #np.savetxt("results/storage_p_profiles.csv", self.storage_p_profiles[:,node_idx,:], delimiter=",")
            #np.savetxt("results/storage_e_profiles.csv", self.storage_e_profiles[:,node_idx,:], delimiter=",")

        print("Energy capacities: ", self.storage_e_capacities_ifft)
        #print("Power capacities: ", self.storage_p_capacities)

        return None
    
    def _determine_cheapest_storage(self):
        #################### Use storage_nodes to check if a storage is located at that node (location in array IDX)
        ############### Use IDX to find the costs in storage_costs
        nodal_costs = NP_FLOAT_MAX * np.ones(self.nodes, dtype=np.float64)
        nodal_n_permutations = np.zeros(self.nodes, dtype=np.int64)

        for node_idx in range(self.nodes): 
            node_mask = self.storage_nodes == node_idx
            node_storage_ids = self.storage_ids[node_mask]

            storage_permutations = frequency.generate_permutations(node_storage_ids)
            nodal_n_permutations[node_idx] = storage_permutations.shape[0]

            for p in range(nodal_n_permutations[node_idx]):
                storage_e_profile_ifft = self.storage_e_profiles_ifft[:,node_idx,:].copy()
                perm_cost = 0
                permutation_storage_p_capacities = np.zeros(len(node_storage_ids), dtype=np.int64)
                permutation_storage_costs = np.zeros((self.storage_costs.shape[0],len(node_storage_ids)), dtype=np.float64)
                permutation_storage_d_efficiencies = np.zeros(len(node_storage_ids), dtype=np.float64)
                permutation_storage_c_efficiencies = np.zeros(len(node_storage_ids), dtype=np.float64)

                for i in range(len(node_storage_ids)):
                    #### This should be fixed to reflect self.storage_ids in case some storage are not included in a scenario
                    permutation_storage_p_capacities[i] = self.CPHP[storage_permutations[p,i]]
                    permutation_storage_costs[:,i] = self.storage_costs[:,storage_permutations[p,i]]
                    #permutation_storage_d_efficiencies = self.storage_d_efficiencies[i]
                    #permutation_storage_c_efficiencies = self.storage_c_efficiencies[i]
                    permutation_storage_d_efficiencies[i] = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], dtype=np.float64)[storage_permutations[p,i]] #### TEST
                    permutation_storage_c_efficiencies[i] = np.array([0.8,0.8,0.8,0.8,0.8,0.9,0.9,0.9,0.9,0.9], dtype=np.float64)[storage_permutations[p,i]] ##### TEST

                storage_e_profile_option, storage_p_profile_option = frequency.reapportion_exceeded_capacity(storage_e_profile_ifft, 
                                                                                                             self.storage_e_capacities_ifft[node_idx,:], 
                                                                                                             permutation_storage_p_capacities, 
                                                                                                             permutation_storage_d_efficiencies, 
                                                                                                             permutation_storage_c_efficiencies, 
                                                                                                             self.resolution)
                
                permutation_annual_discharge = frequency.sum_positive_values(storage_p_profile_option) * self.resolution / self.years
                #print("Annual discharge: ", permutation_annual_discharge)

                perm_cost = np.array([
                                    annualisation_component(
                                        power_capacity=permutation_storage_p_capacities[idx],
                                        energy_capacity=self.storage_e_capacities[idx],
                                        annual_generation=permutation_annual_discharge[idx],
                                        capex_p=permutation_storage_costs[0,idx],
                                        fom=permutation_storage_costs[2,idx],
                                        vom=permutation_storage_costs[3,idx],
                                        lifetime=permutation_storage_costs[4,idx],
                                        discount_rate=permutation_storage_costs[5,idx]
                                    ) for idx in range(0,len(permutation_storage_p_capacities))
                                    ], dtype=np.float64).sum()

                if perm_cost < nodal_costs[node_idx]:
                    nodal_costs[node_idx] = perm_cost

                    for i in range(len(node_storage_ids)):
                        #print(self.storage_e_profiles.shape,storage_e_profile_option.shape,self.storage_p_profiles.shape,storage_p_profile_option.shape)
                        self.storage_e_profiles[:,storage_permutations[p,i]] = storage_e_profile_option[:,i]
                        self.storage_p_profiles[:,storage_permutations[p,i]] = storage_p_profile_option[:,i]
                        self.storage_e_capacities[storage_permutations[p,i]] = self.storage_e_capacities_ifft[node_idx,i]
                        #self.storage_p_capacities[storage_permutations[p,i]] = self.CPHP ##### Rearrange back to CPHP order

        print(nodal_costs)
        storage_costs = sum(nodal_costs)
        return storage_costs

    def _objective(self) -> List[float]:
        deficit = self._reliability(flexible=np.zeros((self.intervals, self.nodes), dtype=np.float64))      
        self.GFlexible_nodal = deficit * self.resolution / self.years / (0.5 * (1 + self.efficiency))

        deficit = self._reliability(flexible=self.GFlexible_nodal)
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution - self.allowance) * 1000000 # MWh
        self.TFlowsAbs = np.abs(self.TFlows)

        self._get_storage_capacities()
        storage_costs = self._determine_cheapest_storage()

        self._apportion_nodal_generation()
        self._calculate_annual_generation()
        cost = self._calculate_costs()
        cost += storage_costs

        loss = self.TFlowsAbs.sum(axis=0) * self.TLoss
        loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - loss)
        
        print("LCOE: ", lcoe, pen_deficit)
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
                    nodes_with_storage,
                    max_frequency,
                    storage_durations,
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
                    storage_e_W_idx,
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
                                storage_unit_types,
                                nodes_with_storage,
                                max_frequency,
                                storage_durations,
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
                                storage_e_W_idx,
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
                storage_unit_types,
                nodes_with_storage,
                max_frequency,
                storage_durations,
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
                storage_e_W_idx,
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
                                storage_unit_types,
                                nodes_with_storage,
                                max_frequency,
                                storage_durations,
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
                                storage_e_W_idx,
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
                                networksteps,)
    solution.evaluate()
    return solution.lcoe + solution.penalties