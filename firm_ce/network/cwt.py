from scipy.fft import fft, ifft
import numpy as np

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
def cwt_peak_detection(signal, scales=np.arange(1, 64, 2)):
    '''Based on Continuos Wavelet Transform with Mexican Hat wavelet'''
    '''https://www.bioconductor.org/packages/devel/bioc/manuals/MassSpecWavelet/man/MassSpecWavelet.pdf'''
    '''https://academic.oup.com/bioinformatics/article/22/17/2059/274284'''
    '''https://pmc.ncbi.nlm.nih.gov/articles/PMC2631518/'''
    '''https://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf'''

    cwt_matrix, scales = get_cwt_matrix(signal, scales)
    local_maxima = get_local_maxima_per_scale(cwt_matrix, scales)
    ridge_list = link_ridges(local_maxima, scales)
    peaks = pick_peaks(ridge_list, cwt_matrix, scales)
    
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

    left = 0
    right = add_length

    total_rows = original_length + left + right
    extended_arr_1d = np.empty((total_rows), dtype=arr_1d.dtype)

    for i in range(original_length):
        extended_arr_1d[left + i] = arr_1d[i]
    
    if right > 0:
        for i in range(right):
            extended_arr_1d[left + original_length + i] = arr_1d[original_length - 1 - i]

    return extended_arr_1d

@njit
def extend_n_base(arr_1d, base=2):    
    original_length = arr_1d.shape[0]
    extended_length = next_power_of_base(original_length, base)
    
    if original_length != extended_length:
        arr_1d = extend_length(arr_1d, add_length=extended_length - original_length)
    
    return arr_1d

@njit 
def get_wavelets(signal_length, scales, wavelet_xlimit=8, wavelet_length=1024):    
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
    daughter_wavelets, scales, len_daughter_wavelets = get_wavelets(original_signal_length, scales)

    signal = extend_n_base(signal)
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
    local_max = np.zeros_like(arr_1d, dtype=np.int32)
    half_window = window_size // 2
    
    for i in range(half_window, len(arr_1d) - half_window):
        if arr_1d[i] == np.max(arr_1d[i - half_window:i + half_window + 1]):
            local_max[i] = 1
    
    max_indices = np.where(local_max > 0)[0]
    if len(max_indices) > 1:
        diffs = np.diff(max_indices)
        to_remove = np.where(diffs < window_size)[0]
        for idx in to_remove:
            if arr_1d[max_indices[idx]] <= arr_1d[max_indices[idx + 1]]:
                local_max[max_indices[idx]] = 0
            else:
                local_max[max_indices[idx + 1]] = 0
    
    return local_max

@njit
def get_local_maxima_per_scale(cwt_matrix, scales, min_window_size=5, amplitude_threshold=0):
    rows, cols = cwt_matrix.shape
    local_maxima = np.zeros(cwt_matrix.shape, dtype=np.int32)
    
    for i, scale in enumerate(scales):
        window_size = max(scale * 2 + 1, min_window_size)
        local_maxima[i,:] = find_local_maximum(cwt_matrix[i,:], window_size)
    
    for i in range(rows):
        for j in range(cols):
            if cwt_matrix[i, j] < amplitude_threshold:
                local_maxima[i, j] = 0
    
    return local_maxima

@njit
def link_ridges(local_maxima, scales, step_direction=-1, final_row_index=0, minimum_window_size=5, gap_threshold=3):
    num_rows = local_maxima.shape[0]
    num_frequencies = local_maxima.shape[1]

    current_max_freq_indices = np.where(local_maxima[num_rows - 1, :] > 0)[0]

    if num_rows > 1:
        row_indices = np.arange(num_rows + step_direction, final_row_index - step_direction, step_direction, dtype=np.int64)
    else:
        row_indices = np.array([0], dtype=np.int64)

    max_possible_ridges = len(current_max_freq_indices) * len(row_indices)

    current_ridge_freq = current_max_freq_indices.astype(np.int32)
    ridge_list = np.full((len(row_indices), max_possible_ridges), -1, dtype=np.int32)
    gap_counter = np.zeros(max_possible_ridges, dtype=np.int32)
    ridges_to_remove = np.full(max_possible_ridges, -1, dtype=np.int32)
    removal_count = 0

    ridge_list[0, :] = current_max_freq_indices
    current_ridge_count = len(current_max_freq_indices)

    for row_iter, row_idx in enumerate(row_indices):
        # Initialise a search window
        current_window_size = max(int(scales[row_idx] * 2 + 1), minimum_window_size)
        
        selected_peak_indices = np.full(current_ridge_count, -1, dtype=np.int32)

        for ridge in range(current_ridge_count):
            # Identify candidate peaks within the search window
            window_start = max(0, current_ridge_freq[ridge] - current_window_size)
            candidate_indices = np.where(local_maxima[row_idx, window_start:min(num_frequencies, current_ridge_freq[ridge] + current_window_size)] > 0)[0]

            candidate_indices += window_start

            if len(candidate_indices) == 0:
                # Mark ridge for removal if no candidate peaks found and gap exceeds threshold
                if gap_counter[ridge] > gap_threshold and scales[row_idx] >= 2:
                    ridges_to_remove[removal_count] = ridge
                    removal_count += 1
                    continue
                # If gap threshold is not exceeded, increment the gap counter but keep ridge
                else:
                    candidate_indices = np.array([current_ridge_freq[ridge]], dtype=np.int32)
                    gap_counter[ridge] += 1
            else:
                # If candidate peaks found, keep ridge and reset gap counter
                gap_counter[ridge] = 0
                best_diff = -1
                for i, cand in enumerate(candidate_indices):
                    diff = abs(candidate_indices[i] - current_ridge_freq[ridge])
                    if diff < best_diff:
                        best_diff = diff
                        best_candidate = candidate_indices[i]
                candidate_indices = np.array([best_candidate], dtype=np.int32)

            ridge_list[row_iter, ridge] = candidate_indices[0]
            selected_peak_indices[ridge] = candidate_indices[0]
            current_ridge_freq[ridge] = candidate_indices[0]

        # Get all candidate peaks for current row
        selected_valid = np.array([spi for spi in selected_peak_indices if spi != -1], dtype=np.int32)

        # Determine which peaks were not used to extend an existing ridge. Then, start new ridges
        unselected_peaks = set_difference_int(np.where(local_maxima[row_idx, :] > 0)[0], selected_valid)

        for candidate_peak in unselected_peaks:
            if current_ridge_count < max_possible_ridges:
                current_ridge_freq[current_ridge_count] = candidate_peak
                for r in range(len(ridge_list)):
                    ridge_list[r, current_ridge_count] = -1
                ridge_list[row_iter, current_ridge_count] = candidate_peak
                gap_counter[current_ridge_count] = 0
                current_ridge_count += 1

        # Remove peaks marked for removal
        if removal_count > 0:
            
            keep_mask = np.ones(current_ridge_count, dtype=np.bool_)
            for i in range(removal_count): # I think this can be made better but not yet sure how
                if ridges_to_remove[i] < current_ridge_count:
                    keep_mask[ridges_to_remove[i]] = False
            
            current_ridge_freq = current_ridge_freq[keep_mask]
            gap_counter = gap_counter[keep_mask]
            ridge_list = ridge_list[:, keep_mask]
            
            current_ridge_count = int(keep_mask.sum())
            removal_count = 0
            max_possible_ridges = len(current_ridge_freq)
        
        current_max_freq_indices = np.concatenate((selected_valid, unselected_peaks))
        
    # Remove empty rows
    keep_mask = np.array([r>0 for r in ridge_list[0]])
    
    return ridge_list[:, keep_mask]

@njit
def pick_peaks(ridge_list, cwt_matrix, scales,
               snr_threshold=2, peak_scale_range=5,
                ridge_length=32, win_size_noise=500, min_noise_level=0.001,
                exclude_boundaries_size=0):
    num_frequencies = cwt_matrix.shape[1]
    
    max_scale = scales[0]
    for i in range(scales.shape[0]):
        max_scale = max(max_scale, scales[i])
    ridge_length = max(ridge_length, max_scale)

    # Determine which scales are valid for peak detection
    # count_valid_scales = int((scales>=peak_scale_range).sum())
    valid_scales = np.array([scale for scale in scales if scale>=peak_scale_range], dtype=np.int32)
    
    # Adjust the minimum noise level based on largest CWT coefficient
    if min_noise_level < 1:
        max_coef = cwt_matrix[0, 0]
        for i in range(len(cwt_matrix)):
            for j in range(cwt_matrix.shape[1]):
                max_coef = max(max_coef, cwt_matrix[i, j])
                    
        min_noise_level = max_coef * min_noise_level

    # Prepare ridges
    num_ridges = ridge_list.shape[1]
    
    #check if this actually sticks
    ridge_list = ridge_list[::-1]

    ridge_lengths = np.array([(ridge>0).sum() for ridge in ridge_list.T], dtype=np.int32)
    # count_valid_ridges = int(sum([ridge[0]>0 for ridge in ridge_list.T]))
    freq_indices = np.array([ridge[0] for ridge in ridge_list.T if ridge[0]>0], dtype=np.int32)

    # Reorder ridges based on frequency index
    order = np.argsort(freq_indices)
    freq_indices = freq_indices[order]
    ridge_list = ridge_list[:, order]
    ridge_lengths = ridge_lengths[order]
    ridge_start_levels = np.zeros(ridge_list.shape[1], dtype=np.int32)

    # Determine peak characteristics for each ridge
    peak_scales = np.empty(num_ridges, dtype=scales.dtype)
    peak_center_indices = np.empty(num_ridges, dtype=np.int32)
    peak_values = np.empty(num_ridges, dtype=cwt_matrix.dtype)
    for ridge in range(num_ridges):
        current_ridge_length = ridge_lengths[ridge]

        # If the ridge has no valid entries, mark with default values.
        if current_ridge_length <= 0:
            peak_scales[ridge] = np.nan
            peak_center_indices[ridge] = -1
            peak_values[ridge] = 0
            continue
        
        # Extract the valid portion of the ridge and scales corresponding to valid ridge points
        ## Does this need to be a copy? or would this do:    
        ridge_freq_indices = ridge_list[:current_ridge_length, ridge]
        scales_for_ridge = scales[:current_ridge_length] 
        

        
        # Select positions within the peak scale range
        selected_flags = np.zeros(current_ridge_length, dtype=np.bool_)
        for j in range(current_ridge_length):
            for vscale in valid_scales:
                if scales_for_ridge[j] == vscale:
                    selected_flags[j] = True
                    break
        # not sure which of theseis actually faster so have left alternative above
        # selected_flags = np.array([(scale==valid_scales).any() for scale in scales_for_ridge])
        
        if int(selected_flags.sum()) == 0:
            peak_scales[ridge] = scales_for_ridge[0]
            peak_center_indices[ridge] = ridge_freq_indices[0]
            peak_values[ridge] = 0
            continue
        
        effective_levels = np.where(selected_flags)[0].astype(np.int32)
        effective_scales = scales_for_ridge[selected_flags]
        effective_freq_indices = ridge_freq_indices[selected_flags]
        
        # Extract CWT coefficients from effective ridge positions, find max coefficient
        ridge_coeff_values = np.array([cwt_matrix[i, j] for i, j in zip(effective_levels, effective_freq_indices)], dtype=cwt_matrix.dtype)
        
        max_val = ridge_coeff_values[0]
        max_idx = 0
        for j, val in enumerate(ridge_coeff_values):
            if val > max_val:
                max_val = val
                max_idx = j
        peak_scales[ridge] = effective_scales[max_idx]
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
                    np.abs(cwt_matrix[col_index_for_noise, :]), ## This is a row not a col?
                    max(0, pci - win_size_noise)
                    )[max(0, pci - win_size_noise):min(num_frequencies, pci + win_size_noise + 1)]
                )
            ) for pci in peak_center_indices]) 

    selected_peaks = freq_indices[
        # select_criterion1 
        np.array([(rl > 0 and rl - 1 < len(scales) and scales[rl-1] >= ridge_length) for rl in ridge_lengths]) * 
        # select_criterion2
        (peak_SNR > snr_threshold) * 
        # select_criterion3 
        np.array([(freq >= exclude_boundaries_size and freq < num_frequencies - exclude_boundaries_size) for freq in freq_indices])
        ]

    return selected_peaks
