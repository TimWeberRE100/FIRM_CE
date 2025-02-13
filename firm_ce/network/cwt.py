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
    
    """ np.savetxt("results/peak_mask.csv", peak_mask, delimiter=",") """
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

    max_possible_ridges = current_max_freq_indices.shape[0] * row_indices.shape[0]

    current_ridge_freq = np.full(max_possible_ridges, -1, dtype=np.int32)  
    ridge_list = np.full((row_indices.shape[0], max_possible_ridges), -1, dtype=np.int32)
    gap_counter = np.zeros(max_possible_ridges, dtype=np.int32)
    ridges_to_remove = np.full(max_possible_ridges, -1, dtype=np.int32)
    removal_count = 0

    current_ridge_count = 0
    for i in range(current_max_freq_indices.shape[0]):
        freq_idx = current_max_freq_indices[i]
        current_ridge_freq[current_ridge_count] = freq_idx
        ridge_list[0, current_ridge_count] = freq_idx
        current_ridge_count += 1

    for row_iter in range(row_indices.shape[0]):
        # Initialise a search window
        current_row = row_indices[row_iter]
        current_scale = scales[current_row]

        computed_window_size = int(current_scale * 2 + 1)
        if computed_window_size < minimum_window_size:
            current_window_size = minimum_window_size
        else:
            current_window_size = computed_window_size

        selected_peak_indices = np.full(current_ridge_count, -1, dtype=np.int32)

        for ridge in range(current_ridge_count):
            # Identify candidate peaks within the search window
            previous_freq = current_ridge_freq[ridge]
            window_start = previous_freq - current_window_size
            if window_start < 0:
                window_start = 0
            window_end = previous_freq + current_window_size
            if window_end > num_frequencies:
                window_end = num_frequencies
            candidate_indices = np.where(local_maxima[current_row, window_start:window_end] > 0)[0]

            for i in range(candidate_indices.shape[0]):
                candidate_indices[i] = candidate_indices[i] + window_start

            if candidate_indices.shape[0] == 0:
                # Mark ridge for removal if no candidate peaks found and gap exceeds threshold
                if gap_counter[ridge] > gap_threshold and current_scale >= 2:
                    ridges_to_remove[removal_count] = ridge
                    removal_count += 1
                    continue
                # If gap threshold is not exceeded, increment the gap counter but keep ridge
                else:
                    candidate_indices = np.empty(1, dtype=np.int32)
                    candidate_indices[0] = previous_freq
                    gap_counter[ridge] += 1
            else:
                # If candidate peaks found, keep ridge and reset gap counter
                gap_counter[ridge] = 0
                best_candidate = candidate_indices[0]
                best_diff = abs(candidate_indices[0] - previous_freq)
                for i in range(1, candidate_indices.shape[0]):
                    diff = abs(candidate_indices[i] - previous_freq)
                    if diff < best_diff:
                        best_diff = diff
                        best_candidate = candidate_indices[i]
                candidate_indices = np.empty(1, dtype=np.int32)
                candidate_indices[0] = best_candidate

            ridge_list[row_iter, ridge] = candidate_indices[0]
            selected_peak_indices[ridge] = candidate_indices[0]
            current_ridge_freq[ridge] = candidate_indices[0]

        # Get all candidate peaks for current row
        next_row_max_indices = np.where(local_maxima[current_row, :] > 0)[0]
        count_selected = 0
        for i in range(current_ridge_count):
            if selected_peak_indices[i] != -1:
                count_selected += 1
        selected_valid = np.empty(count_selected, dtype=np.int32)
        idx_sel = 0
        for i in range(current_ridge_count):
            if selected_peak_indices[i] != -1:
                selected_valid[idx_sel] = selected_peak_indices[i]
                idx_sel += 1

        # Determine which peaks were not used to extend an existing ridge. Then, start new ridges
        unselected_peaks = set_difference_int(next_row_max_indices, selected_valid)

        for i in range(unselected_peaks.shape[0]):
            candidate_peak = unselected_peaks[i]
            if current_ridge_count < max_possible_ridges:
                current_ridge_freq[current_ridge_count] = candidate_peak
                for r in range(ridge_list.shape[0]):
                    ridge_list[r, current_ridge_count] = -1
                ridge_list[row_iter, current_ridge_count] = candidate_peak
                gap_counter[current_ridge_count] = 0
                current_ridge_count += 1

        # Remove peaks marked for removal
        if removal_count > 0:
            keep_mask = np.empty(current_ridge_count, dtype=np.bool_)
            for i in range(current_ridge_count):
                keep_mask[i] = True
            for i in range(removal_count):
                idx_to_remove = ridges_to_remove[i]
                if idx_to_remove < current_ridge_count:
                    keep_mask[idx_to_remove] = False
            
            new_ridge_count = 0
            for i in range(current_ridge_count):
                if keep_mask[i]:
                    new_ridge_count += 1

            new_current_ridge_freq = np.empty(new_ridge_count, dtype=np.int32)
            new_gap_counter = np.empty(new_ridge_count, dtype=np.int32)
            new_ridge_list = np.empty((ridge_list.shape[0], new_ridge_count), dtype=np.int32)
            j = 0
            for i in range(current_ridge_count):
                if keep_mask[i]:
                    new_current_ridge_freq[j] = current_ridge_freq[i]
                    new_gap_counter[j] = gap_counter[i]
                    for r in range(ridge_list.shape[0]):
                        new_ridge_list[r, j] = ridge_list[r, i]
                    j += 1
            current_ridge_freq = new_current_ridge_freq
            gap_counter = new_gap_counter
            ridge_list = new_ridge_list
            current_ridge_count = new_ridge_count
            removal_count = 0
            max_possible_ridges = current_ridge_freq.shape[0]
        
        # Manually concatenate selected and unselected peaks in order to update current_max_freq_indices
        new_length = selected_valid.shape[0] + unselected_peaks.shape[0]
        new_max_freq_indices = np.empty(new_length, dtype=np.int32)
        for i in range(selected_valid.shape[0]):
            new_max_freq_indices[i] = selected_valid[i]
        for i in range(unselected_peaks.shape[0]):
            new_max_freq_indices[selected_valid.shape[0] + i] = unselected_peaks[i]
        current_max_freq_indices = new_max_freq_indices

    # Remove empty rows
    keep_flag = np.empty(ridge_list.shape[1], dtype=np.bool_)
    for ridge in range(ridge_list.shape[1]):
        if ridge_list[0,ridge] > 0:
            keep_flag[ridge] = True
        else:
            keep_flag[ridge] = False

    return ridge_list[:, keep_flag]

@njit
def pick_peaks(ridge_list, cwt_matrix, scales,
               snr_threshold=2, peak_scale_range=5,
                ridge_length=32, win_size_noise=500, min_noise_level=0.001,
                exclude_boundaries_size=0):
    num_scales = cwt_matrix.shape[0]
    num_frequencies = cwt_matrix.shape[1]

    max_scale = scales[0]
    for i in range(scales.shape[0]):
        if scales[i] > max_scale:
            max_scale = scales[i]
    if ridge_length > max_scale:
        ridge_length = max_scale

    # Determine which scales are valid for peak detection
    count_valid_scales = 0
    for i in range(scales.shape[0]):
        if scales[i] >= peak_scale_range:
            count_valid_scales += 1
    valid_scales = np.empty(count_valid_scales, dtype=scales.dtype)
    idx_valid = 0
    for i in range(scales.shape[0]):
        if scales[i] >= peak_scale_range:
            valid_scales[idx_valid] = scales[i]
            idx_valid += 1
    
    # Adjust the minimum noise level based on largest CWT coefficient
    if min_noise_level < 1:
        max_coef = cwt_matrix[0, 0]
        for i in range(num_scales):
            for j in range(num_frequencies):
                if cwt_matrix[i, j] > max_coef:
                    max_coef = cwt_matrix[i, j]
        min_noise_level = max_coef * min_noise_level

    # Prepare ridges
    num_ridges = ridge_list.shape[1]
    ridge_lengths = np.empty(num_ridges, dtype=np.int32)
    ridge_list_copy = ridge_list[::-1].copy() # Flip along axis 0 - SHOULD FIX FIND_RIDGES TO AVOID NEEDING THIS STEP ALTOGETHER
    ridge_list = ridge_list_copy 

    for i in range(num_ridges):
        ridge = ridge_list[:,i]
        ridge_mask = ridge > 0 # Exclude the -1 filler values
        ridge_lengths[i] = len(ridge[ridge_mask])

    count_valid_ridges = 0
    for i in range(num_ridges):
        if ridge_list[0, i] > 0:
            count_valid_ridges += 1
    freq_indices = np.empty(count_valid_ridges, dtype=np.int32)
    idx_freq = 0
    for i in range(num_ridges):
        if ridge_list[0, i] > 0:
            freq_indices[idx_freq] = ridge_list[0, i]
            idx_freq += 1

    ridge_start_levels = np.empty(num_ridges, dtype=np.int32)
    for i in range(num_ridges):
        ridge_start_levels[i] = 0

    # Reorder ridges based on frequency index
    order = np.argsort(freq_indices)
    sorted_freq_indices = np.empty(freq_indices.shape[0], dtype=np.int32)
    for i in range(freq_indices.shape[0]):
        sorted_freq_indices[i] = freq_indices[order[i]]

    sorted_ridge_list = np.empty(ridge_list.shape, dtype=ridge_list.dtype)
    sorted_ridge_lengths = np.empty(num_ridges, dtype=np.int32)
    sorted_ridge_start_levels = np.empty(num_ridges, dtype=np.int32)
    
    for i in range(num_ridges):
        for j in range(ridge_list.shape[0]):
            sorted_ridge_list[j, i] = ridge_list[j, order[i]]
        sorted_ridge_lengths[i] = ridge_lengths[order[i]]
        sorted_ridge_start_levels[i] = ridge_start_levels[order[i]]
    for i in range(sorted_freq_indices.shape[0]):
        freq_indices[i] = sorted_freq_indices[i]
    for i in range(ridge_list.shape[0]):
        for j in range(num_ridges):
            ridge_list[i, j] = sorted_ridge_list[i, j]
    for i in range(num_ridges):
        ridge_lengths[i] = sorted_ridge_lengths[i]
        ridge_start_levels[i] = sorted_ridge_start_levels[i]

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
        ridge_freq_indices = np.empty(current_ridge_length, dtype=np.int32)
        for j in range(current_ridge_length):
            ridge_freq_indices[j] = ridge_list[j, ridge]

        levels = np.empty(current_ridge_length, dtype=np.int32)
        for j in range(current_ridge_length):
            levels[j] = ridge_start_levels[ridge] + j

        scales_for_ridge = np.empty(current_ridge_length, dtype=scales.dtype)
        for j in range(current_ridge_length):
            scales_for_ridge[j] = scales[levels[j]]
        
        # Select positions within the peak scale range
        count_selected = 0
        selected_flags = np.empty(current_ridge_length, dtype=np.bool_)
        for j in range(current_ridge_length):
            flag = False
            for k in range(valid_scales.shape[0]):
                if scales_for_ridge[j] == valid_scales[k]:
                    flag = True
                    break
            selected_flags[j] = flag
            if flag:
                count_selected += 1

        if count_selected == 0:
            peak_scales[ridge] = scales_for_ridge[0]
            peak_center_indices[ridge] = ridge_freq_indices[0]
            peak_values[ridge] = 0
            continue
        
        # Filter the arrays based on the selected flags.
        effective_levels = np.empty(count_selected, dtype=np.int32)
        effective_scales = np.empty(count_selected, dtype=scales.dtype)
        effective_freq_indices = np.empty(count_selected, dtype=np.int32)
        idx_eff = 0
        for j in range(current_ridge_length):
            if selected_flags[j]:
                effective_levels[idx_eff] = levels[j]
                effective_scales[idx_eff] = scales_for_ridge[j]
                effective_freq_indices[idx_eff] = ridge_freq_indices[j]
                idx_eff += 1

        # Extract CWT coefficients from effective ridge positions, find max coefficient
        num_effective = count_selected
        ridge_coeff_values = np.empty(num_effective, dtype=cwt_matrix.dtype)
        for j in range(num_effective):
            freq_idx = effective_freq_indices[j]
            level_idx = effective_levels[j]
            ridge_coeff_values[j] = cwt_matrix[level_idx, freq_idx]
        
        max_val = ridge_coeff_values[0]
        max_idx = 0
        for j in range(1, num_effective):
            if ridge_coeff_values[j] > max_val:
                max_val = ridge_coeff_values[j]
                max_idx = j
        peak_scales[ridge] = effective_scales[max_idx]
        peak_center_indices[ridge] = effective_freq_indices[max_idx]
        peak_values[ridge] = ridge_coeff_values[max_idx]
    
    # Estimate noise and calulate signal-to-noise ratio
    col_index_for_noise = 0
    for i in range(scales.shape[0]):
        if scales[i] == 1:
            col_index_for_noise = i
            break
    noise = np.empty(num_frequencies, dtype=cwt_matrix.dtype)
    for i in range(num_frequencies):
        noise[i] = abs(cwt_matrix[col_index_for_noise, i])
    
    peak_SNR = np.empty(num_ridges, dtype=cwt_matrix.dtype)
    for ridge in range(num_ridges):
        freq_center = peak_center_indices[ridge]
        
        window_start = freq_center - win_size_noise
        if window_start < 0:
            window_start = 0
        window_end = freq_center + win_size_noise + 1
        if window_end > num_frequencies:
            window_end = num_frequencies
        window_size = window_end - window_start
        noise_window = np.empty(window_size, dtype=cwt_matrix.dtype)
        for i in range(window_size):
            noise_window[i] = noise[window_start + i]
        noise_level = quantile_95(noise_window)
        if noise_level < min_noise_level:
            noise_level = min_noise_level
        if noise_level == 0:
            peak_SNR[ridge] = np.inf
        else:
            peak_SNR[ridge] = peak_values[ridge] / noise_level

    # Apply peak selection rules    
    select_criterion1 = np.empty(num_ridges, dtype=np.bool_)
    for i in range(num_ridges):
        if ridge_lengths[i] > 0:
            level_idx = ridge_lengths[i] - 1
            if level_idx < scales.shape[0] and scales[level_idx] >= ridge_length:
                select_criterion1[i] = True
            else:
                select_criterion1[i] = False
        else:
            select_criterion1[i] = False
    
    select_criterion2 = np.empty(num_ridges, dtype=np.bool_)
    for i in range(num_ridges):
        if peak_SNR[i] > snr_threshold:
            select_criterion2[i] = True
        else:
            select_criterion2[i] = False
    
    select_criterion3 = np.empty(freq_indices.shape[0], dtype=np.bool_)
    for i in range(freq_indices.shape[0]):
        freq_val = freq_indices[i]
        if freq_val < exclude_boundaries_size or freq_val >= num_frequencies - exclude_boundaries_size:
            select_criterion3[i] = False
        else:
            select_criterion3[i] = True
    
    final_selection = np.empty(num_ridges, dtype=np.bool_)
    for i in range(num_ridges):
        flag = select_criterion1[i] and select_criterion2[i]
        if i < freq_indices.shape[0]:
            flag = flag and select_criterion3[i]
        final_selection[i] = flag

    selected_peaks = freq_indices[final_selection]
    
    return selected_peaks
