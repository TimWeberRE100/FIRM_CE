# Make Scipy FFT compatible with Numba (no import required): https://github.com/styfenschaer/rocket-fft
import numpy as np
from scipy.fft import rfft, irfft
from time import sleep

from firm_ce.constants import EPSILON_FLOAT64, JIT_ENABLED
from firm_ce.helpers import swap, factorial, max_along_axis_n

if JIT_ENABLED:
    from numba import njit

    # Rocket-FFT does not work for numpy.fft.rfftfreq
    @njit
    def rfftfreq(n, d=1.0):
        val = 1.0 / (n * d)
        N = n // 2 + 1
        freqs = np.arange(0, N, dtype=np.int64)
        return freqs * val
else:
    from scipy.fft import rfftfreq

    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit
def get_frequency_profile(timeseries_profile):
    frequency_profile = rfft(timeseries_profile)
    """ np.savetxt("results/timeseries.csv", timeseries_profile, delimiter=",")    
    np.savetxt("results/frequency.csv", frequency_profile, delimiter=",")  """   
    return frequency_profile

@njit
def convert_to_frequency_magnitudes(frequency_profile):
    magnitudes = np.abs(frequency_profile)
    magnitudes[0] = 0.0 # Remove DC offset
    """ np.savetxt("results/magnitudes.csv", magnitudes, delimiter=",") """
    return magnitudes

@njit
def get_normalised_profile(timeseries_profile):
    magnitudes = convert_to_frequency_magnitudes(timeseries_profile)

    if np.max(magnitudes) == 0:
        return magnitudes
 
    normalised_frequency_profile = magnitudes / np.max(magnitudes)
    """ np.savetxt("results/normalised_magnitudes.csv", normalised_frequency_profile, delimiter=",") """
    return normalised_frequency_profile

@njit
def get_dc_offset(frequency_profile):
    dc_offset = frequency_profile.copy()
    dc_offset[1:] = 0
    """ np.savetxt("results/dc_offset.csv", dc_offset, delimiter=",")  """ 

    return dc_offset

@njit
def get_frequencies(intervals, resolution):
    frequencies = rfftfreq(intervals, d=resolution)
    """ np.savetxt("results/frequencies.csv", frequencies, delimiter=",") """
    return frequencies

@njit
def get_bandpass_filter(lower_cutoff, upper_cutoff, frequencies):
    bandpass_profile = np.zeros(frequencies.shape, dtype=np.float64)

    for idx in range(len(frequencies)):
        if frequencies[idx] > lower_cutoff and frequencies[idx] <= upper_cutoff:
            bandpass_profile[idx] = 1.0
        if frequencies[idx] > upper_cutoff:
            break
    """ np.savetxt(f"results/frequency_{upper_cutoff}_{lower_cutoff}.csv", bandpass_profile, delimiter=",") """
    return bandpass_profile

@njit
def get_magnitude_filter(lower_cutoff, upper_cutoff, normalised_magnitudes):
    n_frequencies = len(normalised_magnitudes)
    filter_profile = np.zeros(n_frequencies, dtype=np.float64)

    for idx in range(1,n_frequencies): # Skip DC offset
        if (normalised_magnitudes[idx] > lower_cutoff - EPSILON_FLOAT64) and (normalised_magnitudes[idx] <= upper_cutoff + EPSILON_FLOAT64):
            filter_profile[idx] = 1.0
            normalised_magnitudes[idx] = -1.0 # Remove this point from being considered in other filters

            # Add harmonics to the filter
            n_harmonics = 2*n_frequencies // idx # Include first set of aliased harmonics
            for n in range(2,n_harmonics):
                harmonic_idx = n*idx % n_frequencies # Take remainder to manage aliased harmonics
                test_idx = n*idx
                if (normalised_magnitudes[(test_idx+1) % n_frequencies] > normalised_magnitudes[harmonic_idx]): 
                    harmonic_idx = (test_idx+1) % n_frequencies
                if (normalised_magnitudes[(test_idx-1) % n_frequencies] > normalised_magnitudes[harmonic_idx]):
                    harmonic_idx = (test_idx-1) % n_frequencies  
                if (normalised_magnitudes[harmonic_idx] > -1 * EPSILON_FLOAT64) and (harmonic_idx > 0):            
                    filter_profile[harmonic_idx] = 1.0
                    normalised_magnitudes[harmonic_idx] = -1.0

    """ np.savetxt(f"results/filter_{upper_cutoff}_{lower_cutoff}.csv", filter_profile, delimiter=",") """
    return filter_profile

@njit
def get_filtered_frequency(frequency_profile, filter_profile, save=False):
    filtered_frequency_profile = frequency_profile * filter_profile

    """ if save:
        max_freq = 0
        for i in range(len(bandpass_filter_profile)):
            if abs(bandpass_filter_profile[i]) > EPSILON_FLOAT64:
                max_freq = i
        np.savetxt(f"results/frequency_filtered_{max_freq}.csv", filtered_frequency_profile, delimiter=",") """
    return filtered_frequency_profile

@njit
def get_timeseries_profile(frequency_profile):
    timeseries_profile = irfft(frequency_profile)

    """ max_freq = 0
    for i in range(len(frequency_profile)):
        if frequency_profile[i] > 0.00001:
            max_freq = i

    print(max_freq)
    np.savetxt(f"results/timeseries_filtered_{max_freq}.csv", timeseries_profile, delimiter=",") """

    return timeseries_profile

@njit
def generate_permutations_impl(a, start, permutations, index):
    n = a.shape[0]
    if start == n - 1:
        for j in range(n):
            permutations[index[0], j] = a[j]
        index[0] += 1
    else:
        for i in range(start, n):
            swap(a, start, i)
            generate_permutations_impl(a, start + 1, permutations, index)
            swap(a, start, i)

@njit
def generate_permutations(array_1d):
    n = len(array_1d)
    n_permutations = factorial(n)

    permutations = np.zeros((n_permutations, n), dtype=array_1d.dtype)

    index = np.zeros(1, dtype=np.int64)
    a = array_1d.copy()

    generate_permutations_impl(a, 0, permutations, index)
    
    return permutations

@njit
def apportion_nodal_noise(nodal_timeseries, noise_timeseries):
    intervals, nodal_balancing = nodal_timeseries.shape

    # Get initial power constraints
    soft_power_constraints = max_along_axis_n(np.abs(nodal_timeseries), 0)
    
    for i in range(intervals):
        for b in range(nodal_balancing):
            #reverse_b = nodal_balancing - b - 1 # Apportion from highest frequency balancing technology down to lowest
            noise_sum = noise_timeseries[i] + nodal_timeseries[i, b]

            if noise_sum > soft_power_constraints[b]:            
                nodal_timeseries[i, b] = soft_power_constraints[b]
                noise_timeseries[i] = noise_sum - soft_power_constraints[b]
            elif noise_sum < -1 * soft_power_constraints[b]:
                nodal_timeseries[i, b] = -1 * soft_power_constraints[b]
                noise_timeseries[i] = noise_sum + soft_power_constraints[b]
            else:
                nodal_timeseries[i, b] = noise_sum
                noise_timeseries[i] = 0

    if np.sum(np.abs(noise_timeseries)) > 0:
        nodal_timeseries[:,0] = nodal_timeseries[:,0] + noise_timeseries

    return nodal_timeseries

""" @njit
def precharge_storage(interval, 
                        nodal_e_capacities,
                        nodal_p_timeseries_profiles, 
                        nodal_e_timeseries_profiles, 
                        balancing_d_efficiencies, 
                        balancing_c_efficiencies, 
                        balancing_d_constraints, 
                        balancing_c_constraints, 
                        time_resolution):
    
    precharge = True
    pre_interval = interval - 1
    
    if pre_interval < 1:
        precharge = False

    while precharge:
        # Determine which storages are empty

        # Check previous interval

        # Try to prefill empty storages using other storages

        #

        

        if pre_interval < 1:
            precharge = False
        elif np.sum(nodal_e_timeseries_profiles[pre_interval,:]) - np.sum(nodal_e_capacities) < EPSILON_FLOAT64: 
            ### NEED TO LIMIT TO STORAGE SYSTEMS CAPACITIES
            precharge = False
        else:
            pre_interval -= 1 """

@njit
def order_balancing(node_balancing_order, node_balancing_e_capacities, variable_costs_per_mwh):
    n = node_balancing_order.shape[0]
    indices = np.arange(n)

    for i in range(n):
        for j in range(i + 1, n):
            # Compare energy capacity first
            if (node_balancing_e_capacities[indices[j]] > node_balancing_e_capacities[indices[i]]) or \
               (node_balancing_e_capacities[indices[j]] == node_balancing_e_capacities[indices[i]] and
                variable_costs_per_mwh[indices[j]] > variable_costs_per_mwh[indices[i]]):
                # Swap for descending order
                temp = indices[i]
                indices[i] = indices[j]
                indices[j] = temp

    # Create the node permutation based on the sorted indices
    node_permutation = np.empty_like(node_balancing_order, dtype=np.int64)
    for i in range(n):
        node_permutation[i] = node_balancing_order[indices[i]]

    return node_permutation

""" @njit
def apply_balancing_constraints(nodal_p_timeseries_profiles, 
                                  nodal_e_capacities, 
                                  balancing_d_efficiencies, 
                                  balancing_c_efficiencies, 
                                  balancing_d_constraints, 
                                  balancing_c_constraints, 
                                  time_resolution):
    
    intervals, storage_number = nodal_p_timeseries_profiles.shape
    nodal_e_timeseries_profiles = np.zeros(nodal_p_timeseries_profiles.shape, dtype=np.float64)
    nodal_capacities_mwh = 1000 * nodal_e_capacities
    nodal_capacities_d_mw = 1000 * balancing_d_constraints
    nodal_capacities_c_mw = 1000 * balancing_c_constraints
    nodal_deficit = np.zeros(intervals, dtype=np.float64)
    nodal_spillage = np.zeros(intervals, dtype=np.float64)

    #print(nodal_capacities_mwh.shape,nodal_e_timeseries_profiles.shape)
    
    nodal_e_timeseries_profiles[0,:] = 0.5 * nodal_capacities_mwh

    for interval in range(intervals):       
        if interval < intervals-1:
            storage_t_1 = nodal_e_timeseries_profiles[interval,:]
            energy_t_1 = -1 * nodal_p_timeseries_profiles[interval, :] * time_resolution
            for n in range(storage_number):
                storage_t_1[n] += energy_t_1[n] * balancing_d_efficiencies[n] if nodal_p_timeseries_profiles[interval, n] > 0 else energy_t_1[n] * balancing_c_efficiencies[n]

        print(f"Initial INTERVAL {interval}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
        for n in range(storage_number-1):
            # Check for discharging overflow
            if (storage_t_1[n] < 0.0) or (nodal_p_timeseries_profiles[interval, n] > nodal_capacities_d_mw[n]):
                excess_e = min(storage_t_1[n],
                                  (nodal_capacities_d_mw[n] - nodal_p_timeseries_profiles[interval, n]) * balancing_d_efficiencies[n] * time_resolution)
                storage_t_1[n] -= excess_e

                excess_p = min(excess_e / balancing_d_efficiencies[n] / time_resolution,
                                  (nodal_capacities_d_mw[n] - nodal_p_timeseries_profiles[interval, n]))                
                nodal_p_timeseries_profiles[interval, n] += excess_p                    

                if interval < intervals-1:
                    storage_t_1[n+1] += excess_p * balancing_d_efficiencies[n+1] * time_resolution                       
                nodal_p_timeseries_profiles[interval, n+1] -= excess_p

                print(f"Up discharging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
                #print("Excesses: ", excess_p, excess_e, storage_d_efficiencies[n], storage_d_efficiencies[n+1])
                
            # Check for charging overflow
            elif (storage_t_1[n] > nodal_capacities_mwh[n]) or (nodal_p_timeseries_profiles[interval, n] < nodal_capacities_c_mw[n]):
                deficit_e = min(nodal_capacities_mwh[n] - storage_t_1[n],
                                   (nodal_p_timeseries_profiles[interval, n] - nodal_capacities_c_mw[n]) * balancing_c_efficiencies[n] * time_resolution)
                storage_t_1[n] += deficit_e

                deficit_p = min(deficit_e / balancing_c_efficiencies[n] / time_resolution,
                                   (nodal_p_timeseries_profiles[interval, n] - nodal_capacities_c_mw[n]))
                nodal_p_timeseries_profiles[interval, n] -= deficit_p                    

                if interval < intervals-1:
                    storage_t_1[n+1] -= deficit_p * balancing_c_efficiencies[n+1] * time_resolution
                nodal_p_timeseries_profiles[interval, n+1] += deficit_p 

                print(f"Up charging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
                exit()

        # Reverse direction of apportioning
        for n in range(storage_number-1):
            k = storage_number - n - 1
                
            # Check for discharging overflow
            if (storage_t_1[k] < 0.0) or (nodal_p_timeseries_profiles[interval, k] > nodal_capacities_d_mw[k]):
                excess_e = min(storage_t_1[k],
                                 (nodal_capacities_d_mw[k] - nodal_p_timeseries_profiles[interval, k]) * balancing_d_efficiencies[k] * time_resolution)
                storage_t_1[k] -= excess_e

                excess_p = min(excess_e / balancing_d_efficiencies[k] / time_resolution,
                                  (nodal_capacities_d_mw[k] - nodal_p_timeseries_profiles[interval, k]))
                nodal_p_timeseries_profiles[interval, k] += excess_p                    

                if interval < intervals-1:
                    storage_t_1[k-1] += excess_p * balancing_d_efficiencies[k-1] * time_resolution                    
                nodal_p_timeseries_profiles[interval, k-1] -= excess_p 

                print(f"Down discharging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)

            # Check for charging overflow
            elif (storage_t_1[k] > nodal_capacities_mwh[k]) or (nodal_p_timeseries_profiles[interval, k] < nodal_capacities_c_mw[k]):
                deficit_e = min((nodal_capacities_mwh[k] - storage_t_1[k]),
                                   (nodal_p_timeseries_profiles[interval, k] - nodal_capacities_c_mw[k]) * balancing_c_efficiencies[k] * time_resolution)
                storage_t_1[k] += deficit_e

                deficit_p = min(deficit_e / balancing_c_efficiencies[k] / time_resolution,
                                   (nodal_p_timeseries_profiles[interval, k] - nodal_capacities_c_mw[k]))
                nodal_p_timeseries_profiles[interval, k] -= deficit_p                    

                if interval < intervals-1:
                    storage_t_1[k-1] -= deficit_p * balancing_c_efficiencies[k-1] * time_resolution
                nodal_p_timeseries_profiles[interval, k-1] += deficit_p

                print(f"Down charing deficits: ", deficit_p, deficit_e)
                print(f"Down charging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
                exit()

        # Determine deficits
        if (storage_t_1[0] < 0.0) or (nodal_p_timeseries_profiles[interval, 0] > nodal_capacities_d_mw[0]):
            excess_e = min(storage_t_1[0],
                             (nodal_capacities_d_mw[0] - nodal_p_timeseries_profiles[interval, 0]) * balancing_d_efficiencies[0] * time_resolution) 
            storage_t_1[0] -= excess_e

            excess_p = min(excess_e / balancing_d_efficiencies[0] / time_resolution,
                              (nodal_capacities_d_mw[0] - nodal_p_timeseries_profiles[interval, 0]))
            nodal_p_timeseries_profiles[interval, 0] += excess_p                

            nodal_deficit[interval] = np.abs(excess_p)

            print(f"Deficit discharging: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)

        elif (storage_t_1[0] > nodal_capacities_mwh[0]) or (nodal_p_timeseries_profiles[interval, 0] < nodal_capacities_c_mw[0]):
            deficit_e = min((nodal_capacities_mwh[0] - storage_t_1[0]),
                                (nodal_p_timeseries_profiles[interval, 0] - nodal_capacities_c_mw[0]) * balancing_c_efficiencies[0] * time_resolution)
            storage_t_1[0] += deficit_e

            deficit_p = min(deficit_e / balancing_c_efficiencies[0] / time_resolution,
                               (nodal_p_timeseries_profiles[interval, 0] - nodal_capacities_c_mw[0]))
            nodal_p_timeseries_profiles[interval, 0] -= deficit_p               

            nodal_spillage[interval] = deficit_p

            print(f"Deficit charging: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
        #sleep(1)
        
        if interval < intervals-1:
            nodal_e_timeseries_profiles[interval+1, :] = storage_t_1

    return nodal_p_timeseries_profiles, nodal_e_timeseries_profiles, nodal_deficit, nodal_spillage """

@njit
def apply_balancing_constraints(nodal_p_timeseries_profiles, 
                                  balancing_d_constraints, 
                                  balancing_c_constraints):
    
    intervals, storage_number = nodal_p_timeseries_profiles.shape
    nodal_capacities_d_mw = 1000 * balancing_d_constraints
    nodal_capacities_c_mw = 1000 * balancing_c_constraints
    nodal_deficit = np.zeros(intervals, dtype=np.float64)
    nodal_spillage = np.zeros(intervals, dtype=np.float64)

    for interval in range(intervals):        
        """ print(f"Initial INTERVAL {interval}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """
        for n in range(storage_number-1):
            # Check for discharging overflow
            if (nodal_p_timeseries_profiles[interval, n] > nodal_capacities_d_mw[n]):
                excess_p = nodal_capacities_d_mw[n] - nodal_p_timeseries_profiles[interval, n]              
                nodal_p_timeseries_profiles[interval, n] += excess_p                    

                nodal_p_timeseries_profiles[interval, n+1] -= excess_p

                """ print(f"Up discharging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """
                #print("Excesses: ", excess_p, excess_e, storage_d_efficiencies[n], storage_d_efficiencies[n+1])
                
            # Check for charging overflow
            elif (nodal_p_timeseries_profiles[interval, n] < nodal_capacities_c_mw[n]):
                deficit_p = nodal_p_timeseries_profiles[interval, n] - nodal_capacities_c_mw[n]
                nodal_p_timeseries_profiles[interval, n] -= deficit_p                    

                nodal_p_timeseries_profiles[interval, n+1] += deficit_p 

                """ print(f"Up charging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
                exit() """

        # Reverse direction of apportioning
        for n in range(storage_number-1):
            k = storage_number - n - 1
                
            # Check for discharging overflow
            if (nodal_p_timeseries_profiles[interval, k] > nodal_capacities_d_mw[k]):
                excess_p = nodal_capacities_d_mw[k] - nodal_p_timeseries_profiles[interval, k]
                nodal_p_timeseries_profiles[interval, k] += excess_p                    

                nodal_p_timeseries_profiles[interval, k-1] -= excess_p 

                """ print(f"Down discharging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """

            # Check for charging overflow
            elif (nodal_p_timeseries_profiles[interval, k] < nodal_capacities_c_mw[k]):
                deficit_p = nodal_p_timeseries_profiles[interval, k] - nodal_capacities_c_mw[k]
                nodal_p_timeseries_profiles[interval, k] -= deficit_p                    

                nodal_p_timeseries_profiles[interval, k-1] += deficit_p

                """ print(f"Down charing deficits: ", deficit_p, deficit_e)
                print(f"Down charging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
                exit() """

        # Determine deficits
        if (nodal_p_timeseries_profiles[interval, 0] > nodal_capacities_d_mw[0]):
            excess_p = nodal_capacities_d_mw[0] - nodal_p_timeseries_profiles[interval, 0]
            nodal_p_timeseries_profiles[interval, 0] += excess_p                

            nodal_deficit[interval] = np.abs(excess_p)

            """ print(f"Deficit discharging: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """

        elif (nodal_p_timeseries_profiles[interval, 0] < nodal_capacities_c_mw[0]):
            deficit_p = nodal_p_timeseries_profiles[interval, 0] - nodal_capacities_c_mw[0]
            nodal_p_timeseries_profiles[interval, 0] -= deficit_p               

            nodal_spillage[interval] = deficit_p

            """ print(f"Deficit charging: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """
        #sleep(1)
        
    return nodal_p_timeseries_profiles, nodal_deficit, nodal_spillage