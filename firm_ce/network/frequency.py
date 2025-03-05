# Make Scipy FFT compatible with Numba (no import required): https://github.com/styfenschaer/rocket-fft
import numpy as np
from scipy.fft import rfft, irfft

from firm_ce.constants import JIT_ENABLED
from firm_ce.helpers import max_along_axis_n

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
    return frequency_profile

@njit
def convert_to_frequency_magnitudes(frequency_profile):
    magnitudes = np.abs(frequency_profile)
    magnitudes[0] = 0.0 # Remove DC offset
    return magnitudes

@njit
def get_normalised_profile(timeseries_profile):
    magnitudes = convert_to_frequency_magnitudes(timeseries_profile)

    if np.max(magnitudes) == 0:
        return magnitudes
 
    normalised_frequency_profile = magnitudes / np.max(magnitudes)
    return normalised_frequency_profile

@njit
def get_dc_offset(frequency_profile):
    dc_offset = frequency_profile.copy()
    dc_offset[1:] = 0
    return dc_offset

@njit
def get_frequencies(intervals, resolution):
    frequencies = rfftfreq(intervals, d=resolution)
    return frequencies

@njit
def get_bandpass_filter(lower_cutoff, upper_cutoff, frequencies):
    bandpass_profile = np.zeros(frequencies.shape, dtype=np.float64)

    for idx in range(len(frequencies)):
        if frequencies[idx] > lower_cutoff and frequencies[idx] <= upper_cutoff:
            bandpass_profile[idx] = 1.0
        if frequencies[idx] > upper_cutoff:
            break
    return bandpass_profile

@njit
def get_filtered_frequency(frequency_profile, filter_profile):
    filtered_frequency_profile = frequency_profile * filter_profile
    return filtered_frequency_profile

@njit
def get_timeseries_profile(frequency_profile):
    timeseries_profile = irfft(frequency_profile)
    return timeseries_profile

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

@njit
def apply_balancing_constraints(nodal_p_timeseries_profiles, 
                                  nodal_e_capacities, 
                                  balancing_d_efficiencies, 
                                  balancing_c_efficiencies, 
                                  balancing_d_constraints, 
                                  balancing_c_constraints, 
                                  time_resolution):
    
    # INTERVALS ARE WRONG FOR STORED ENERGY
    intervals, storage_number = nodal_p_timeseries_profiles.shape

    nodal_e_timeseries_profiles = np.zeros((intervals, storage_number), dtype=np.float64)
    nodal_capacities_mwh = 1000.0 * nodal_e_capacities
    nodal_capacities_d_mw = 1000.0 * balancing_d_constraints
    nodal_capacities_c_mw = 1000.0 * balancing_c_constraints

    nodal_deficit = np.zeros(intervals, dtype=np.float64)
    nodal_spillage = np.zeros(intervals, dtype=np.float64)

    for j in range(storage_number):
        nodal_e_timeseries_profiles[0, j] = 0.5 * nodal_capacities_mwh[j]

    for t in range(intervals):
        non_final_t = (t < intervals - 1)
        storage = nodal_e_timeseries_profiles[t, :]
        p_profile = nodal_p_timeseries_profiles[t, :]

        if non_final_t:
            energy = -p_profile * time_resolution
            for i in range(storage_number):
                if p_profile[i] > 0.0:
                    storage[i] += energy[i] * balancing_d_efficiencies[i]
                else:
                    storage[i] += energy[i] * balancing_c_efficiencies[i]

        for n in range(storage_number - 1):
            # Discharging overflow check:
            if (storage[n] < 0.0) or (p_profile[n] > nodal_capacities_d_mw[n]):
                excess_e = storage[n]
                limit_e = (nodal_capacities_d_mw[n] - p_profile[n]) * balancing_d_efficiencies[n] * time_resolution
                if excess_e > limit_e:
                    excess_e = limit_e
                storage[n] -= excess_e

                excess_p = excess_e / (balancing_d_efficiencies[n] * time_resolution)
                limit_p = nodal_capacities_d_mw[n] - p_profile[n]
                if excess_p > limit_p:
                    excess_p = limit_p
                p_profile[n] += excess_p
                if non_final_t:
                    storage[n+1] += excess_p * balancing_d_efficiencies[n+1] * time_resolution
                p_profile[n+1] -= excess_p

            # Charging overflow check:
            elif (storage[n] > nodal_capacities_mwh[n]) or (p_profile[n] < nodal_capacities_c_mw[n]):
                deficit_e = nodal_capacities_mwh[n] - storage[n]
                limit_e = (p_profile[n] - nodal_capacities_c_mw[n]) * balancing_c_efficiencies[n] * time_resolution
                if deficit_e > limit_e:
                    deficit_e = limit_e
                storage[n] += deficit_e

                deficit_p = deficit_e / (balancing_c_efficiencies[n] * time_resolution)
                limit_p = p_profile[n] - nodal_capacities_c_mw[n]
                if deficit_p > limit_p:
                    deficit_p = limit_p
                p_profile[n] -= deficit_p
                if non_final_t:
                    storage[n+1] -= deficit_p * balancing_c_efficiencies[n+1] * time_resolution
                p_profile[n+1] += deficit_p

        for n in range(storage_number - 1):
            k = storage_number - n - 1
            # Discharging overflow check:
            if (storage[k] < 0.0) or (p_profile[k] > nodal_capacities_d_mw[k]):
                excess_e = storage[k]
                limit_e = (nodal_capacities_d_mw[k] - p_profile[k]) * balancing_d_efficiencies[k] * time_resolution
                if excess_e > limit_e:
                    excess_e = limit_e
                storage[k] -= excess_e

                excess_p = excess_e / (balancing_d_efficiencies[k] * time_resolution)
                limit_p = nodal_capacities_d_mw[k] - p_profile[k]
                if excess_p > limit_p:
                    excess_p = limit_p
                p_profile[k] += excess_p
                if non_final_t:
                    storage[k-1] += excess_p * balancing_d_efficiencies[k-1] * time_resolution
                p_profile[k-1] -= excess_p

            # Charging overflow check:
            elif (storage[k] > nodal_capacities_mwh[k]) or (p_profile[k] < nodal_capacities_c_mw[k]):
                deficit_e = nodal_capacities_mwh[k] - storage[k]
                limit_e = (p_profile[k] - nodal_capacities_c_mw[k]) * balancing_c_efficiencies[k] * time_resolution
                if deficit_e > limit_e:
                    deficit_e = limit_e
                storage[k] += deficit_e

                deficit_p = deficit_e / (balancing_c_efficiencies[k] * time_resolution)
                limit_p = p_profile[k] - nodal_capacities_c_mw[k]
                if deficit_p > limit_p:
                    deficit_p = limit_p
                p_profile[k] -= deficit_p
                if non_final_t:
                    storage[k-1] -= deficit_p * balancing_c_efficiencies[k-1] * time_resolution
                p_profile[k-1] += deficit_p

        # Final check at the first storage unit; record deficit/spillage
        if (storage[0] < 0.0) or (p_profile[0] > nodal_capacities_d_mw[0]):
            excess_e = storage[0]
            limit_e = (nodal_capacities_d_mw[0] - p_profile[0]) * balancing_d_efficiencies[0] * time_resolution
            if excess_e > limit_e:
                excess_e = limit_e
            storage[0] -= excess_e
            excess_p = excess_e / (balancing_d_efficiencies[0] * time_resolution)
            limit_p = nodal_capacities_d_mw[0] - p_profile[0]
            if excess_p > limit_p:
                excess_p = limit_p
            p_profile[0] += excess_p
            nodal_deficit[t] = np.abs(excess_p)
        elif (storage[0] > nodal_capacities_mwh[0]) or (p_profile[0] < nodal_capacities_c_mw[0]):
            deficit_e = nodal_capacities_mwh[0] - storage[0]
            limit_e = (p_profile[0] - nodal_capacities_c_mw[0]) * balancing_c_efficiencies[0] * time_resolution
            if deficit_e > limit_e:
                deficit_e = limit_e
            storage[0] += deficit_e
            deficit_p = deficit_e / (balancing_c_efficiencies[0] * time_resolution)
            limit_p = p_profile[0] - nodal_capacities_c_mw[0]
            if deficit_p > limit_p:
                deficit_p = limit_p
            p_profile[0] -= deficit_p
            nodal_spillage[t] = deficit_p

        # Propagate the updated storage to the next interval
        if non_final_t:
            nodal_e_timeseries_profiles[t+1, :] = storage

    return nodal_p_timeseries_profiles
