from scipy.fft import rfft, irfft, rfftfreq
#import matplotlib.pyplot as plt
import numpy as np

from firm_ce.constants import EPSILON_FLOAT64

def get_frequency_profile(timeseries_profile):
    frequency_profile = rfft(timeseries_profile)
    """ np.savetxt("results/timeseries.csv", timeseries_profile, delimiter=",")    
    np.savetxt("results/frequency.csv", frequency_profile, delimiter=",")   """  
    return frequency_profile

def convert_to_frequency_magnitudes(frequency_profile):
    magnitudes = np.abs(frequency_profile)
    magnitudes[0] = 0.0 # Remove DC offset
    """ np.savetxt("results/magnitudes.csv", magnitudes, delimiter=",") """
    return magnitudes

def get_normalised_profile(timeseries_profile):
    magnitudes = convert_to_frequency_magnitudes(timeseries_profile)

    if np.max(magnitudes) == 0:
        return magnitudes
 
    normalised_frequency_profile = magnitudes / np.max(magnitudes)
    """ np.savetxt("results/normalised_magnitudes.csv", normalised_frequency_profile, delimiter=",") """
    return normalised_frequency_profile

def get_dc_offset(frequency_profile):
    dc_offset = frequency_profile.copy()
    dc_offset[1:] = 0
    """ np.savetxt("results/dc_offset.csv", dc_offset, delimiter=",")  """ 

    return dc_offset

def get_frequencies(intervals, resolution):
    return rfftfreq(intervals, d=resolution)

def get_bandpass_filter(lower_cutoff, upper_cutoff, frequencies):
    bandpass_profile = np.zeros(frequencies.shape, dtype=np.float64)

    """ for idx in range(len(frequencies)):
        if frequencies[idx] > lower_cutoff and frequencies[idx] <= upper_cutoff:
            bandpass_profile[idx] = 1.0
        if frequencies[idx] > upper_cutoff:
            break
    np.savetxt(f"results/frequency_{upper_cutoff}_{lower_cutoff}.csv", bandpass_profile, delimiter=",") """
    return bandpass_profile

 
def get_filtered_frequency(frequency_profile, bandpass_filter_profile, save=False):
    filtered_frequency_profile = frequency_profile * bandpass_filter_profile

    """ if save:
        max_freq = 0
        for i in range(len(bandpass_filter_profile)):
            if abs(bandpass_filter_profile[i]) > EPSILON_FLOAT64:
                max_freq = i
        np.savetxt(f"results/frequency_filtered_{max_freq}.csv", filtered_frequency_profile, delimiter=",") """
    return filtered_frequency_profile


def get_timeseries_profile(frequency_profile):
    timeseries_profile = irfft(frequency_profile)

    max_freq = 0
    for i in range(len(frequency_profile)):
        if frequency_profile[i] > 0.00001:
            max_freq = i

    """ print(max_freq)
    np.savetxt(f"results/timeseries_filtered_{max_freq}.csv", timeseries_profile, delimiter=",") """

    return timeseries_profile

def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

def generate_permutations(array_1d):
    n = len(array_1d)
    n_permutations = factorial(n)

    if n == 1:
        return array_1d

    permutations = np.zeros((n_permutations, n), dtype=array_1d.dtype)

    def permute(array_1d, index, current_perm):
        if len(array_1d) == 0:
            permutations[index[0]] = current_perm
            index[0] += 1
            return
        
        for i in range(len(array_1d)):
            permute(np.concatenate((array_1d[:i], array_1d[i+1:])), index, np.append(current_perm, array_1d[i]))

    permute(array_1d, [0], np.array([], dtype=array_1d.dtype))
    return permutations

def reapportion_exceeded_capacity(nodal_e_timeseries_profiles,nodal_e_capacities, time_resolution):
    intervals, storage_number = nodal_e_timeseries_profiles.shape
    nodal_p_timeseries_profiles = np.zeros(nodal_e_timeseries_profiles.shape, dtype=np.float64)
    nodal_capacities_mwh = 1000 * nodal_e_capacities

    for interval in range(intervals):
        changed = True
        while changed:
            changed = False
            print(interval, nodal_e_timeseries_profiles[interval, :], nodal_capacities_mwh, nodal_p_timeseries_profiles[interval-1,:])
            for n in range(storage_number):
                # Check for positive overflow:
                if (nodal_e_timeseries_profiles[interval, n] > nodal_capacities_mwh[n] + 0.001):
                    excess = nodal_e_timeseries_profiles[interval, n] - nodal_capacities_mwh[n]
                    nodal_e_timeseries_profiles[interval, n] -= excess

                    # If not at the end, add the excess to the next node;
                    # otherwise, add it to the previous node.
                    ###### THIS SHOULD REAPPORTION BASED UPON THE CHEAPEST POWER CAPACITY ARRANGEMENT?
                    if n < storage_number - 1:
                        nodal_e_timeseries_profiles[interval, n+1] += excess                        
                    elif n > 0: ####### NEED TO MAKE THIS FLIP AND MOVE ALL THE WAY BACK TO FIRST NODE
                        nodal_e_timeseries_profiles[interval, n-1] += excess
                    changed = True

                # Check for negative overflow
                elif (nodal_e_timeseries_profiles[interval, n] < -0.001):
                    deficit = nodal_e_timeseries_profiles[interval, n] 
                    nodal_e_timeseries_profiles[interval, n] -= deficit
                    
                    # If not at the end, add the deficit to the next node;
                    # otherwise, add it to the previous node.
                    if n < storage_number - 1:
                        nodal_e_timeseries_profiles[interval, n+1] += deficit
                    elif n > 0:
                        nodal_e_timeseries_profiles[interval, n-1] += deficit
                    changed = True
        
        if interval > 0:
            nodal_p_timeseries_profiles[interval-1, :] = -1 * (nodal_e_timeseries_profiles[interval, :] - nodal_e_timeseries_profiles[interval-1, :]) / time_resolution

    return nodal_p_timeseries_profiles, nodal_e_timeseries_profiles