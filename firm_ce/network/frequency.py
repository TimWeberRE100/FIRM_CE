# Make Scipy FFT compatible with Numba (no import required): https://github.com/styfenschaer/rocket-fft
import numpy as np
from scipy.fft import rfft, irfft

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
    np.savetxt("results/frequency.csv", frequency_profile, delimiter=",")     """
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
def get_filtered_frequency(frequency_profile, bandpass_filter_profile, save=False):
    filtered_frequency_profile = frequency_profile * bandpass_filter_profile

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
def reapportion_exceeded_capacity(nodal_e_timeseries_profiles, nodal_e_capacities, storage_d_efficiencies, storage_c_efficiencies, time_resolution):
    intervals, storage_number = nodal_e_timeseries_profiles.shape
    nodal_p_timeseries_profiles = np.zeros(nodal_e_timeseries_profiles.shape, dtype=np.float64)
    nodal_capacities_mwh = 1000 * nodal_e_capacities
    #nodal_capacities_mw = 1000 * nodal_p_capacities

    for interval in range(intervals):
        changed = True
        
        while changed:
            changed = False
            if interval > 0:
                power_t_1 = (nodal_e_timeseries_profiles[interval-1, :] - nodal_e_timeseries_profiles[interval, :]) / time_resolution
                for n in range(storage_number):
                    power_t_1[n] = power_t_1[n] / storage_d_efficiencies[n] if power_t_1[n] > 0 else power_t_1[n] / storage_c_efficiencies[n]
            else:
                power_t_1 = np.zeros(storage_number, dtype=np.float64)

            #print(interval, nodal_e_timeseries_profiles[interval, :], nodal_capacities_mwh, power_t_1)
            for n in range(storage_number):
                # Check for energy capacity positive overflow
                if (nodal_e_timeseries_profiles[interval, n] > nodal_capacities_mwh[n] + 0.001):
                    excess_e = nodal_e_timeseries_profiles[interval, n] - nodal_capacities_mwh[n]
                    nodal_e_timeseries_profiles[interval, n] -= excess_e

                    excess_p = excess_e / storage_c_efficiencies[n]
                    power_t_1 -= excess_p

                    # If not at the end, add the excess to the next node;
                    # otherwise, add it to the previous node.
                    if n < storage_number - 1:
                        nodal_e_timeseries_profiles[interval, n+1] += excess_e
                        if interval > 0:
                            power_t_1[n+1] += excess_e / storage_c_efficiencies[n+1]                     
                    elif n > 0:
                        nodal_e_timeseries_profiles[interval, n-1] += excess_e #### NEED TO REVERSE AND HEAD ALL THE WAY BACK
                        if interval > 0:
                            power_t_1[n-1] += excess_e / storage_c_efficiencies[n-1]
                    changed = True

                # Check for energy capacity negative overflow
                elif (nodal_e_timeseries_profiles[interval, n] < -0.001):
                    deficit_e = nodal_e_timeseries_profiles[interval, n] 
                    nodal_e_timeseries_profiles[interval, n] -= deficit_e

                    deficit_p = deficit_e / storage_d_efficiencies[n]
                    power_t_1 -= deficit_p
                    
                    # If not at the end, add the deficit to the next node;
                    # otherwise, add it to the previous node.
                    if n < storage_number - 1:
                        nodal_e_timeseries_profiles[interval, n+1] += deficit_e
                        if interval > 0:
                            power_t_1[n+1] += deficit_e / storage_d_efficiencies[n+1]
                    elif n > 0:
                        nodal_e_timeseries_profiles[interval, n-1] += deficit_e
                        if interval > 0:
                            power_t_1[n-1] += deficit_e / storage_d_efficiencies[n-1]
                    changed = True

                """ # Check for power capacity positive overflow
                elif power_t_1[n] > nodal_capacities_mw[n]:
                    excess_p = power_t_1[n] - nodal_capacities_mw[n]
                    power_t_1[n] -= excess_p

                    excess_e = excess_p * storage_c_efficiencies[n]
                    nodal_e_timeseries_profiles[interval, n] -= excess_e

                    if n < storage_number - 1:
                        nodal_e_timeseries_profiles[interval, n+1] += excess_e
                        if interval > 0:
                            power_t_1[n+1] += excess_e / storage_c_efficiencies[n+1]                     
                    elif n > 0:
                        nodal_e_timeseries_profiles[interval, n-1] += excess_e #### NEED TO REVERSE AND HEAD ALL THE WAY BACK
                        if interval > 0:
                            power_t_1[n-1] += excess_e / storage_c_efficiencies[n-1]
                    changed = True

                # Check for power capacity negative overflow
                elif power_t_1[n] < -1 * nodal_capacities_mw[n]:
                    deficit_p = power_t_1[n] - nodal_capacities_mw[n]
                    power_t_1[n] -= deficit_p

                    deficit_e = deficit_p * storage_d_efficiencies[n]
                    nodal_e_timeseries_profiles[interval, n] -= deficit_e

                    if n < storage_number - 1:
                        nodal_e_timeseries_profiles[interval, n+1] += deficit_e
                        if interval > 0:
                            power_t_1[n+1] += deficit_e / storage_d_efficiencies[n+1]
                    elif n > 0:
                        nodal_e_timeseries_profiles[interval, n-1] += deficit_e
                        if interval > 0:
                            power_t_1[n-1] += deficit_e / storage_d_efficiencies[n-1]
                    changed = True """
        
        if interval > 0:
            nodal_p_timeseries_profiles[interval-1, :] = power_t_1

    return nodal_p_timeseries_profiles, nodal_e_timeseries_profiles

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