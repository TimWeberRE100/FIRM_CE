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

"""
def get_timeseries_profile(frequency_profile):
    return timeseries_profile

def apportion_dc_offset(dc_offset, timeseries_profiles):
    return timeseries_profiles_with_dc """

######## Pass flexible profile

######## Pass storage power profile (negative for charging)

######## Pass stored energy profile