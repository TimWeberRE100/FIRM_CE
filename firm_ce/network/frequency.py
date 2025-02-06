# Make Scipy FFT compatible with Numba (no import required): https://github.com/styfenschaer/rocket-fft
import numpy as np
from scipy.fft import rfft, irfft
from time import sleep

from firm_ce.constants import EPSILON_FLOAT64, JIT_ENABLED

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
    np.savetxt("results/frequency.csv", frequency_profile, delimiter=",") """    
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
    """ np.savetxt("results/dc_offset.csv", dc_offset, delimiter=",")   """

    return dc_offset

@njit
def get_frequencies(intervals, resolution):
    return rfftfreq(intervals, d=resolution)

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
    filter_profile = np.zeros(normalised_magnitudes.shape, dtype=np.float64)

    for idx in range(1,len(normalised_magnitudes)): # Skip DC offset
        if (normalised_magnitudes[idx] > lower_cutoff - EPSILON_FLOAT64) and (normalised_magnitudes[idx] <= upper_cutoff + EPSILON_FLOAT64):
            filter_profile[idx] = 1.0
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
def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

@njit
def swap(a, i, j):
    temp = a[i]
    a[i] = a[j]
    a[j] = temp

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

    """ if n == 1:
        return array_1d """

    permutations = np.zeros((n_permutations, n), dtype=array_1d.dtype)

    index = np.zeros(1, dtype=np.int64)
    a = array_1d.copy()

    generate_permutations_impl(a, 0, permutations, index)

    """ def permute(array_1d, index, current_perm):
        if len(array_1d) == 0:
            permutations[index[0]] = current_perm
            index[0] += 1
            return
        
        for i in range(len(array_1d)):
            permute(np.concatenate((array_1d[:i], array_1d[i+1:])), index, np.append(current_perm, array_1d[i]))

    permute(array_1d, [0], np.array([], dtype=array_1d.dtype)) """
    return permutations

@njit
def __reapportion_exceeded_capacity(nodal_p_timeseries_profiles, nodal_p_capacities, nodal_e_capacities, storage_d_efficiencies, storage_c_efficiencies, time_resolution):
    intervals, storage_number = nodal_p_timeseries_profiles.shape
    nodal_e_timeseries_profiles = np.zeros(nodal_p_timeseries_profiles.shape, dtype=np.float64)
    nodal_capacities_mwh = 1000 * nodal_e_capacities
    nodal_capacities_mw = 1000 * nodal_p_capacities
    node_power_deficit = np.zeros(intervals, dtype=np.float64)
    
    nodal_e_timeseries_profiles[0,:] = 0.5 * nodal_capacities_mwh

    for interval in range(intervals):
        changed = True
        
        while changed: ###### I THINK THE DEFICIT LETS US REMOVE CHANGED?
            changed = False
            if interval < intervals-1:
                storage_t_1 = nodal_e_timeseries_profiles[interval,:]
                energy_t_1 = -1 * nodal_p_timeseries_profiles[interval, :] * time_resolution
                for n in range(storage_number):
                    storage_t_1[n] += energy_t_1[n] * storage_d_efficiencies[n] if nodal_p_timeseries_profiles[interval, n] > 0 else energy_t_1[n] * storage_c_efficiencies[n]

            #print(f"Initial INTERVAL {interval}: ", interval, nodal_p_timeseries_profiles[interval, :], nodal_capacities_mw, storage_t_1, nodal_capacities_mwh)
            for n in range(storage_number-1):
                # Check for discharging overflow
                if (nodal_p_timeseries_profiles[interval, n] > nodal_capacities_mw[n]) or (storage_t_1[n] < 0.0):
                    excess_p = max(nodal_p_timeseries_profiles[interval, n] - nodal_capacities_mw[n], 
                                      -1 * storage_t_1[n] / storage_d_efficiencies[n] / time_resolution)                 

                    nodal_p_timeseries_profiles[interval, n] -= excess_p

                    excess_e = excess_p * storage_d_efficiencies[n] * time_resolution
                    storage_t_1[n] += excess_e

                    if interval < intervals-1:
                        storage_t_1[n+1] -= excess_p * storage_d_efficiencies[n+1] * time_resolution                       
                    nodal_p_timeseries_profiles[interval, n+1] += excess_p

                    #print(f"Up discharging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], nodal_capacities_mw, storage_t_1, nodal_capacities_mwh)
                    #print("Excesses: ", excess_p, excess_e, storage_d_efficiencies[n], storage_d_efficiencies[n+1])
                # Check for charging overflow
                elif (nodal_p_timeseries_profiles[interval, n] < -1 * nodal_capacities_mw[n]) or (storage_t_1[n] > nodal_capacities_mwh[n]):
                    deficit_p = min(nodal_p_timeseries_profiles[interval, n] - (-1 * nodal_capacities_mw[n]),
                                       (nodal_capacities_mwh[n] - storage_t_1[n]) / storage_c_efficiencies[n] / time_resolution)
                    nodal_p_timeseries_profiles[interval, n] -= deficit_p

                    deficit_e = deficit_p * storage_c_efficiencies[n] * time_resolution
                    storage_t_1[n] += deficit_e

                    if interval < intervals-1:
                        storage_t_1[n+1] -= deficit_p * storage_c_efficiencies[n+1] * time_resolution
                    nodal_p_timeseries_profiles[interval, n+1] += deficit_p 

                    #print(f"Up charging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], nodal_capacities_mw, storage_t_1, nodal_capacities_mwh)

            for n in range(storage_number-1):
                k = storage_number - n - 1
                
                # Check for power capacity positive overflow
                if (nodal_p_timeseries_profiles[interval, k] > nodal_capacities_mw[k]) or (storage_t_1[k] < 0.0):
                    excess_p = max(nodal_p_timeseries_profiles[interval, k] - nodal_capacities_mw[k],
                                      -1 * storage_t_1[k] / storage_d_efficiencies[k] / time_resolution)
                    nodal_p_timeseries_profiles[interval, k] -= excess_p

                    excess_e = excess_p * storage_d_efficiencies[k] * time_resolution
                    storage_t_1[k] += excess_e

                    if interval < intervals-1:
                        storage_t_1[k-1] -= excess_p * storage_d_efficiencies[k-1] * time_resolution                        
                    nodal_p_timeseries_profiles[interval, k-1] += excess_p 

                    #print(f"Down discharging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], nodal_capacities_mw, storage_t_1, nodal_capacities_mwh)

                # Check for power capacity negative overflow
                elif (nodal_p_timeseries_profiles[interval, k] < -1 * nodal_capacities_mw[k]) or (storage_t_1[k] > nodal_capacities_mwh[k]):
                    deficit_p = min(nodal_p_timeseries_profiles[interval, k] - (-1 * nodal_capacities_mw[k]),
                                       (nodal_capacities_mwh[k] - storage_t_1[k]) / storage_c_efficiencies[k] / time_resolution)
                    nodal_p_timeseries_profiles[interval, k] -= deficit_p

                    deficit_e = deficit_p * storage_c_efficiencies[k] * time_resolution
                    storage_t_1[k] += deficit_e

                    if interval < intervals-1:
                        storage_t_1[k-1] -= deficit_p * storage_c_efficiencies[k-1] * time_resolution
                    nodal_p_timeseries_profiles[interval, k-1] += deficit_p

                    #print(f"Down charging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], nodal_capacities_mw, storage_t_1, nodal_capacities_mwh)


            # Determine deficits
            if (nodal_p_timeseries_profiles[interval, 0] > nodal_capacities_mw[0]) or (storage_t_1[0] < 0.0):
                excess_p = max(nodal_p_timeseries_profiles[interval, 0] - nodal_capacities_mw[0],
                                  -1 * storage_t_1[0] / storage_d_efficiencies[0] / time_resolution)
                nodal_p_timeseries_profiles[interval, 0] -= excess_p

                excess_e = excess_p * storage_d_efficiencies[0] * time_resolution
                storage_t_1[0] += excess_e

                node_power_deficit[interval] = np.abs(excess_p)

                #print(f"Deficit discharging: ", interval, nodal_p_timeseries_profiles[interval, :], nodal_capacities_mw, storage_t_1, nodal_capacities_mwh)

            elif (nodal_p_timeseries_profiles[interval, 0] < -1 * nodal_capacities_mw[0]) or (storage_t_1[0] > nodal_capacities_mwh[0]):
                deficit_p = min(nodal_p_timeseries_profiles[interval, 0] - (-1 * nodal_capacities_mw[0]),
                                   (nodal_capacities_mwh[0] - storage_t_1[0]) / storage_c_efficiencies[0] / time_resolution)
                nodal_p_timeseries_profiles[interval, 0] -= deficit_p

                deficit_e = deficit_p * storage_c_efficiencies[0] * time_resolution
                storage_t_1[0] += deficit_e

                node_power_deficit[interval] = np.abs(deficit_p)

                #print(f"Deficit charging: ", interval, nodal_p_timeseries_profiles[interval, :], nodal_capacities_mw, storage_t_1, nodal_capacities_mwh)
            #sleep(1)
        
        if interval < intervals-1:
            nodal_e_timeseries_profiles[interval+1, :] = storage_t_1

    return nodal_p_timeseries_profiles, nodal_e_timeseries_profiles, node_power_deficit

@njit
def reapportion_exceeded_capacity(nodal_p_timeseries_profiles, nodal_e_capacities, storage_d_efficiencies, storage_c_efficiencies, time_resolution):
    intervals, storage_number = nodal_p_timeseries_profiles.shape
    nodal_e_timeseries_profiles = np.zeros(nodal_p_timeseries_profiles.shape, dtype=np.float64)
    nodal_capacities_mwh = 1000 * nodal_e_capacities
    node_power_deficit = np.zeros(intervals, dtype=np.float64)
    
    nodal_e_timeseries_profiles[0,:] = 0.5 * nodal_capacities_mwh

    for interval in range(intervals):
        changed = True
        """ if interval > 6800:
            print(storage_d_efficiencies, storage_c_efficiencies)
            exit() """
        
        while changed: ###### I THINK THE DEFICIT LETS US REMOVE CHANGED?
            changed = False
            if interval < intervals-1:
                storage_t_1 = nodal_e_timeseries_profiles[interval,:]
                energy_t_1 = -1 * nodal_p_timeseries_profiles[interval, :] * time_resolution
                for n in range(storage_number):
                    storage_t_1[n] += energy_t_1[n] * storage_d_efficiencies[n] if nodal_p_timeseries_profiles[interval, n] > 0 else energy_t_1[n] * storage_c_efficiencies[n]

            """ print(f"Initial INTERVAL {interval}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """
            for n in range(storage_number-1):
                # Check for discharging overflow
                if (storage_t_1[n] < 0.0):
                    excess_e = storage_t_1[n]
                    storage_t_1[n] -= excess_e

                    excess_p = excess_e / storage_d_efficiencies[n] / time_resolution                
                    nodal_p_timeseries_profiles[interval, n] += excess_p                    

                    if interval < intervals-1:
                        storage_t_1[n+1] += excess_p * storage_d_efficiencies[n+1] * time_resolution                       
                    nodal_p_timeseries_profiles[interval, n+1] -= excess_p

                    """ print(f"Up discharging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """
                    #print("Excesses: ", excess_p, excess_e, storage_d_efficiencies[n], storage_d_efficiencies[n+1])
                
                # Check for charging overflow
                elif (storage_t_1[n] > nodal_capacities_mwh[n]):
                    deficit_e = (nodal_capacities_mwh[n] - storage_t_1[n])
                    storage_t_1[n] += deficit_e

                    deficit_p = deficit_e / storage_c_efficiencies[n] / time_resolution
                    nodal_p_timeseries_profiles[interval, n] -= deficit_p                    

                    if interval < intervals-1:
                        storage_t_1[n+1] -= deficit_p * storage_c_efficiencies[n+1] * time_resolution
                    nodal_p_timeseries_profiles[interval, n+1] += deficit_p 

                    """ print(f"Up charging {n}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
                    exit() """

            # Reverse direction of apportioning
            for n in range(storage_number-1):
                k = storage_number - n - 1
                
                # Check for discharging overflow
                if (storage_t_1[k] < 0.0):
                    excess_e = storage_t_1[k]
                    storage_t_1[k] -= excess_e

                    excess_p = excess_e / storage_d_efficiencies[k] / time_resolution
                    nodal_p_timeseries_profiles[interval, k] += excess_p                    

                    if interval < intervals-1:
                        storage_t_1[k-1] += excess_p * storage_d_efficiencies[k-1] * time_resolution                    
                    nodal_p_timeseries_profiles[interval, k-1] -= excess_p 

                    """ print(f"Down discharging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """

                # Check for charging overflow
                elif (storage_t_1[k] > nodal_capacities_mwh[k]):
                    deficit_e = (nodal_capacities_mwh[k] - storage_t_1[k])
                    storage_t_1[k] += deficit_e

                    deficit_p = deficit_e / storage_c_efficiencies[k] / time_resolution
                    nodal_p_timeseries_profiles[interval, k] -= deficit_p                    

                    if interval < intervals-1:
                        storage_t_1[k-1] -= deficit_p * storage_c_efficiencies[k-1] * time_resolution
                    nodal_p_timeseries_profiles[interval, k-1] += deficit_p

                    """ print(f"Down charing deficits: ", deficit_p, deficit_e)
                    print(f"Down charging {k}: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh)
                    exit() """

            # Determine deficits
            if (storage_t_1[0] < 0.0):
                excess_e = storage_t_1[0] 
                storage_t_1[0] -= excess_e

                excess_p = excess_e / storage_d_efficiencies[0] / time_resolution
                nodal_p_timeseries_profiles[interval, 0] += excess_p                

                node_power_deficit[interval] = np.abs(excess_p)

                """ print(f"Deficit discharging: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """

            elif (storage_t_1[0] > nodal_capacities_mwh[0]):
                deficit_e = (nodal_capacities_mwh[0] - storage_t_1[0]) 
                storage_t_1[0] += deficit_e

                deficit_p = deficit_e / storage_c_efficiencies[0] / time_resolution
                nodal_p_timeseries_profiles[interval, 0] -= deficit_p               

                node_power_deficit[interval] = np.abs(deficit_p)

                """ print(f"Deficit charging: ", interval, nodal_p_timeseries_profiles[interval, :], storage_t_1, nodal_capacities_mwh) """
            #sleep(1)
        
        if interval < intervals-1:
            nodal_e_timeseries_profiles[interval+1, :] = storage_t_1

    return nodal_p_timeseries_profiles, nodal_e_timeseries_profiles, node_power_deficit

@njit
def sum_positive_values(arr):
    rows, cols = arr.shape
    result = np.zeros(cols, dtype=arr.dtype)
    
    for j in range(cols): 
        col_sum = 0
        for i in range(rows):  
            if arr[i, j] > 0:  
                col_sum += arr[i, j]
        result[j] = col_sum 

    return result

@njit
def max_along_axis_n(arr, axis_n):
    rows, cols = arr.shape
    max_vals = np.empty(cols, dtype=arr.dtype)
    for j in range(cols):
        max_vals[j] = arr[axis_n, j]  
        for i in range(1, rows): 
            if arr[i, j] > max_vals[j]:
                max_vals[j] = arr[i, j]
    return max_vals