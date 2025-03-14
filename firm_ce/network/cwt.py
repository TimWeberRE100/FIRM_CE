from scipy.fft import fft, ifft
import numpy as np
import time

from firm_ce.constants import JIT_ENABLED
from firm_ce.helpers import set_difference_int, quantile_95

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit
def cwt_peak_detection(signal):
    '''Based on Continuos Wavelet Transform with Mexican Hat wavelet'''
    '''https://www.bioconductor.org/packages/devel/bioc/manuals/MassSpecWavelet/man/MassSpecWavelet.pdf'''
    '''https://academic.oup.com/bioinformatics/article/22/17/2059/274284'''
    '''https://pmc.ncbi.nlm.nih.gov/articles/PMC2631518/'''
    '''https://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf'''

    scales=np.arange(1, 64, 2)
    cwt_matrix, scales = get_cwt_matrix(signal, scales)
    local_maxima = get_local_maxima_per_scale(cwt_matrix, scales, 9)
    ridge_list, ridge_lengths = link_ridges(local_maxima, scales, -1, 0, 9, 3, 15)
    peaks = pick_peaks(ridge_list, cwt_matrix, scales, ridge_lengths, 2, 5, 500, 0.001)
    
    peak_mask = np.zeros(signal.size, dtype=np.int32)
    noise_mask = np.ones(signal.size, dtype=np.int32)
    for peak in peaks:
        peak_mask[peak] = 1
        noise_mask[peak] = 0
    noise_mask[0] = 0
    
    return peak_mask, noise_mask

@njit
def generate_mexican_hat_wavelet(x):
    # Mother wavelet scaled and translated into daughter wavelets for CWT
    return (2 / np.sqrt(3)) * (np.pi**(-0.25)) * (1 - x**2) * np.exp(-x**2 / 2)

@njit
def next_power_of_base(n, base):
    return base**int(np.ceil(np.log(n) / np.log(base)))

@njit
def extend_length(arr_1d, add_length):   
    original_length = arr_1d.shape[0]    
    total_rows = original_length + add_length
    extended_arr_1d = np.empty((total_rows), dtype=arr_1d.dtype)

    extended_arr_1d[:original_length] = arr_1d    
    extended_arr_1d[original_length:] = arr_1d[original_length - 1 : original_length - 1 - add_length : -1]

    return extended_arr_1d

@njit
def extend_n_base(arr_1d, base):    
    original_length = arr_1d.shape[0]
    extended_length = next_power_of_base(original_length, base)
    
    if original_length != extended_length:
        arr_1d = extend_length(arr_1d, extended_length - original_length)
    
    return arr_1d

@njit 
def get_wavelets(signal_length, scales, wavelet_xlimit, wavelet_length):    
    mother_wavelet_xval = np.linspace(-wavelet_xlimit, wavelet_xlimit, wavelet_length)
    mother_wavelet = generate_mexican_hat_wavelet(mother_wavelet_xval)

    extended_signal_base = max(signal_length, next_power_of_base(signal_length, 2))
    
    mother_wavelet_xval -= mother_wavelet_xval[0]
    dxval = mother_wavelet_xval[1]
    xmax = mother_wavelet_xval[-1]
    
    daughter_wavelets = np.zeros((len(scales), extended_signal_base), dtype=np.complex128)
    len_daughter_wavelets = np.zeros(len(scales), dtype=np.int64)

    for i, scale in enumerate(scales):
        scaled_wavelet_filter = np.zeros(extended_signal_base, dtype=np.float64)
        scaled_wavelet_indices = 1 + np.trunc(np.arange(scale * xmax + 1) / (scale * dxval)).astype(np.int32)
        
        if len(scaled_wavelet_indices) == 1:
            scaled_wavelet_indices = np.array([1, 1])
        
        len_wave = len(scaled_wavelet_indices)
        scaled_wavelet_filter[:len_wave] = np.flip(mother_wavelet[scaled_wavelet_indices - 1]) - np.mean(mother_wavelet[scaled_wavelet_indices - 1])  # Adjust indexing
        
        if len(scaled_wavelet_filter) > extended_signal_base:
            break
        
        len_daughter_wavelets[i] = len_wave
        daughter_wavelets[i,:] = np.conj(fft(scaled_wavelet_filter))

    return daughter_wavelets, scales, len_daughter_wavelets

@njit
def circular_shift(w_coefs_i, shift_idx):
    n = len(w_coefs_i)
    shifted_array = np.empty_like(w_coefs_i)
    
    shifted_array[:n - shift_idx] = w_coefs_i[shift_idx:]
    shifted_array[n - shift_idx:] = w_coefs_i[:shift_idx]
    
    return shifted_array

@njit
def get_cwt_matrix(signal, scales):
    original_signal_length = len(signal)
    daughter_wavelets, scales, len_daughter_wavelets = get_wavelets(original_signal_length, scales, 8, 1024)

    signal = extend_n_base(signal, 2)
    signal_fft = fft(signal)
    extended_signal_length = len(signal)
    cwt_matrix = np.zeros((len(scales), original_signal_length), dtype=np.float64)

    for i, scale_i in enumerate(scales):
        # Convolution via FFT
        w_coefs_i = (1 / np.sqrt(scale_i)) * np.real(ifft(signal_fft * daughter_wavelets[i,:])) / extended_signal_length
        len_wave = len_daughter_wavelets[i]
        
        # Shift the coefficients with half wavelet width
        shift_idx = extended_signal_length - len_wave // 2
        w_coefs_i = circular_shift(w_coefs_i, shift_idx)
        cwt_matrix[i,:] = w_coefs_i[:original_signal_length]

    cwt_matrix = cwt_matrix[:len(scales),:]    
    
    return cwt_matrix, scales

@njit
def find_local_maximum(arr_1d, window_size):
    # Uses a double-ended queue for the sliding window
    n = len(arr_1d)
    local_max = np.zeros(n, dtype=np.bool_)
    half_window = window_size // 2
    
    deque_indices = np.empty(n, dtype=np.int64) # Faster just to make this n elements rather than calc many modulo if it's window_size elements
    head = 0 
    tail = 0  
    last_peak_index = -window_size
    
    for i in range(n):
        if head < tail and deque_indices[head] <= i - window_size:
            head += 1
        while head < tail and arr_1d[deque_indices[tail - 1]] <= arr_1d[i]:
            tail -= 1
        deque_indices[tail] = i
        tail += 1

        if i >= window_size - 1:
            center = i - half_window
            if 0 <= center < n and arr_1d[center] == arr_1d[deque_indices[head]]:
                if center - last_peak_index >= window_size:
                    local_max[center] = True
                    last_peak_index = center
                else:
                    if arr_1d[center] > arr_1d[last_peak_index]:
                        local_max[last_peak_index] = False
                        local_max[center] = True
                        last_peak_index = center

    return local_max

@njit
def get_local_maxima_per_scale(cwt_matrix, scales, min_window_size):
    rows, cols = cwt_matrix.shape
    local_maxima = np.zeros((rows, cols), dtype=np.int32)
    
    for i in range(rows):
        window_size = max(scales[i] * 2 + 1, min_window_size)
        local_maxima[i, :] = find_local_maximum(cwt_matrix[i, :], window_size)
                
    return local_maxima

@njit
def link_ridges(local_maxima, scales, step_direction, final_row_index, minimum_window_size, gap_threshold, min_ridge_length):
    num_rows, num_frequencies = local_maxima.shape
    current_max_freq_indices = np.where(local_maxima[-1, :] > 0)[0]

    if num_rows > 1:
        if step_direction < 0:
            row_indices = np.arange(num_rows - 2, final_row_index - 1, step_direction)
        else:
            row_indices = np.arange(1, final_row_index + 1, step_direction)
    else:
        row_indices = np.zeros(1, dtype=np.int64)

    ridge_list = np.full((len(row_indices) + 1, num_frequencies), -1, dtype=np.int32)
    gap_counter = np.zeros(num_frequencies, dtype=np.int32)
    ridges_to_remove = np.full(num_frequencies, -1, dtype=np.int32)
    current_ridge_freq = np.full(num_frequencies, -1, dtype=np.int32)
    ridge_lengths = np.full(num_frequencies, -1, dtype=np.int32)
    keep_mask = np.full(num_frequencies, False, dtype=np.bool_)
    removal_count = 0

    current_ridge_count = len(current_max_freq_indices)
    ridge_list[0, :current_ridge_count] = current_max_freq_indices.copy()
    keep_mask[:current_ridge_count] = True 
    ridge_lengths[:current_ridge_count] = 1
    current_ridge_freq[:current_ridge_count] = current_max_freq_indices.copy()

    for row_iter, row_idx in enumerate(row_indices):
        row_iter_i = row_iter + 1

        # Initialise a search window
        current_window_size = max(int(scales[row_idx] * 2 + 1), minimum_window_size)        
        selected_peak_indices = np.full(current_ridge_count, -1, dtype=np.int32)

        for ridge in range(current_ridge_count):
            # Identify candidate peaks within the search window
            window_start = max(0, current_ridge_freq[ridge] - current_window_size)
            window_end = min(num_frequencies, current_ridge_freq[ridge] + current_window_size)
            candidate_indices = np.where(local_maxima[row_idx, window_start:window_end] > 0)[0]
            candidate_indices += window_start

            if len(candidate_indices) == 0:
                # Mark ridge for removal if no candidate peaks found and gap exceeds threshold
                if gap_counter[ridge] > gap_threshold:
                    ridges_to_remove[removal_count] = ridge
                    removal_count += 1
                    continue
                # If gap threshold is not exceeded, increment the gap counter but keep ridge
                else:
                    candidate_indices = np.array([current_ridge_freq[ridge]], dtype=np.int32)
                    gap_counter[ridge] += 1
            else:
                # If candidate peaks found, keep ridge and reset gap counter
                if candidate_indices.size >= 2:
                    diffs = np.abs(candidate_indices - current_ridge_freq[ridge])
                    best_index = np.argmin(diffs)
                    candidate = candidate_indices[best_index]
                else:
                    candidate = candidate_indices[0]
                candidate_indices = np.array([candidate], dtype=np.int32)
    
            ridge_list[row_iter_i, ridge] = candidate_indices[0]
            selected_peak_indices[ridge] = candidate_indices[0]
            current_ridge_freq[ridge] = candidate_indices[0]
            ridge_lengths[ridge] += 1

        # Get all candidate peaks for current row
        selected_valid = np.array([spi for spi in selected_peak_indices if spi != -1], dtype=np.int32)

        # Determine which peaks were not used to extend an existing ridge. 
        # Then, start new ridges that are capable of reaching the minimum length
        if row_idx > min_ridge_length:
            unselected_peaks = set_difference_int(np.where(local_maxima[row_idx, :] > 0)[0], selected_valid)
            for candidate_peak in unselected_peaks:
                if current_ridge_count < num_frequencies:
                    ridge_list[row_iter_i, current_ridge_count] = candidate_peak
                    gap_counter[current_ridge_count] = 0
                    keep_mask[current_ridge_count] = True
                    current_ridge_freq[current_ridge_count] = candidate_peak
                    current_ridge_count += 1                

        # Mark peaks for removal
        if removal_count > 0:            
            for i in range(removal_count): 
                if ridges_to_remove[i] < current_ridge_count:
                    keep_mask[ridges_to_remove[i]] = False
            
            current_ridge_count = int(keep_mask.sum())
            removal_count = 0
        
        current_max_freq_indices = np.concatenate((selected_valid, unselected_peaks))
    
    return ridge_list[:, keep_mask], ridge_lengths[keep_mask]

@njit
def pick_peaks(ridge_list, cwt_matrix, scales, ridge_lengths,
               snr_threshold, peak_scale_range,
                win_size_noise, min_noise_level):
    num_frequencies = cwt_matrix.shape[1]

    # Determine which scales are valid for peak detection
    valid_scales = np.array([scale for scale in scales if scale>=peak_scale_range], dtype=np.int32)
    
    # Adjust the minimum noise level based on largest CWT coefficient
    if min_noise_level < 1:
        max_coef = cwt_matrix[0, 0]
        for i in range(len(cwt_matrix)):
            for j in range(cwt_matrix.shape[1]):
                max_coef = max(max_coef, cwt_matrix[i, j])
                    
        min_noise_level = max_coef * min_noise_level

    # Prepare ridges
    ridge_list = ridge_list[::-1]
    ridge_lengths = ridge_lengths[::-1]
    freq_indices = np.array([ridge[0] for ridge in ridge_list.T if ridge[0]>0], dtype=np.int32)

    # Reorder ridges based on frequency index
    order = np.argsort(freq_indices)
    freq_indices = freq_indices[order]
    ridge_list = ridge_list[:, order]
    ridge_lengths = ridge_lengths[order]

    # Determine peak characteristics for each ridge
    peak_center_indices = np.empty(ridge_list.shape[1], dtype=np.int32)
    peak_values = np.empty(ridge_list.shape[1], dtype=cwt_matrix.dtype)
    for ridge in range(ridge_list.shape[1]):
        ridge_freq_indices = ridge_list[:, ridge]
        current_ridge_length = ridge_lengths[ridge]

        # If the ridge has no valid entries, mark with default values.
        if current_ridge_length <= 0:
            peak_center_indices[ridge] = -1
            peak_values[ridge] = 0
            continue
        
        # Extract the valid portion of the ridge and scales corresponding to valid ridge points
        ridge_freq_indices = ridge_list[:current_ridge_length, ridge]
        scales_for_ridge = scales[:current_ridge_length] 
        
        # Select positions within the peak scale range
        selected_flags = np.zeros(current_ridge_length, dtype=np.bool_)
        for j in range(current_ridge_length):
            for vscale in valid_scales:
                if scales_for_ridge[j] == vscale:
                    selected_flags[j] = True
                    break
        
        if int(selected_flags.sum()) == 0:
            peak_center_indices[ridge] = ridge_freq_indices[0]
            peak_values[ridge] = 0
            continue
        
        effective_freq_indices = ridge_freq_indices[selected_flags]
        
        # Extract CWT coefficients from effective ridge positions, find max coefficient
        selected_indices = np.where(selected_flags)[0].astype(np.int32)
        ridge_coeff_values = np.empty(len(selected_indices), dtype=cwt_matrix.dtype)

        for idx in range(len(selected_indices)):
            ridge_coeff_values[idx] = cwt_matrix[selected_indices[idx], effective_freq_indices[idx]]
        
        max_val = ridge_coeff_values[0]
        max_idx = 0
        for j, val in enumerate(ridge_coeff_values):
            if val > max_val:
                max_val = val
                max_idx = j
        peak_center_indices[ridge] = effective_freq_indices[max_idx]
        peak_values[ridge] = ridge_coeff_values[max_idx]
    
    # Estimate noise and calulate signal-to-noise ratio
    col_index_for_noise = 0
    for i, scale in enumerate(scales):
        if scale == 1:
            col_index_for_noise = i
            break
        
    peak_SNR = peak_values / np.array(
        [max(
            min_noise_level, 
            quantile_95(
                np.roll(
                    np.abs(cwt_matrix[col_index_for_noise, :]),
                    max(0, pci - win_size_noise)
                    )[max(0, pci - win_size_noise):min(num_frequencies, pci + win_size_noise + 1)]
                )
            ) for pci in peak_center_indices]) 

    selected_peaks = freq_indices[(peak_SNR > snr_threshold)]

    return selected_peaks
