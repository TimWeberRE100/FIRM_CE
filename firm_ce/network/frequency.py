from scipy.fft import rfft, irfft, rfftfreq
#import matplotlib.pyplot as plt
import numpy as np

def get_frequency_profile(timeseries_profile):
    frequency_profile = rfft(timeseries_profile)    
    return frequency_profile

def convert_to_frequency_magnitudes(frequency_profile):
    magnitudes = np.abs(frequency_profile)
    magnitudes[0] = 0.0 # Remove DC offset
    return magnitudes

def get_normalised_profile(timeseries_profile):
    magnitudes = convert_to_frequency_magnitudes(timeseries_profile)

    if np.max(magnitudes) == 0:
        return magnitudes
 
    normalised_frequency_profile = magnitudes / np.max(magnitudes)
    return normalised_frequency_profile

def get_dc_offset(frequency_profile):
    dc_offset = frequency_profile[0]
    return dc_offset

def get_frequencies(intervals, resolution):
    return rfftfreq(intervals, d=resolution)

def get_bandpass_filter(lower_cutoff, upper_cutoff, frequencies):
    bandpass_profile = np.zeros(frequencies.shape, dtype=np.float64)
    for idx in range(len(frequencies)):
        if frequencies[idx] > lower_cutoff and frequencies[idx] < upper_cutoff:
            bandpass_profile[idx] = 1.0
        if frequencies[idx] > upper_cutoff:
            break

    return bandpass_profile

 
def get_filtered_frequency(frequency_profile, bandpass_filter_profile):
    filtered_frequency_profile = frequency_profile * bandpass_filter_profile
    return filtered_frequency_profile


def get_timeseries_profile(frequency_profile):
    timeseries_profile = irfft(frequency_profile)
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

    
    
    


"""
def apportion_dc_offset(dc_offset, timeseries_profiles):
    return timeseries_profiles_with_dc """

######## Pass flexible profile

######## Pass storage power profile (negative for charging)

######## Pass stored energy profile