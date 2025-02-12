from scipy.fft import fft, ifft
import numpy as np

from firm_ce.constants import JIT_ENABLED

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
def cwt_peak_detection(signal, scales=None):
    '''Based on Continuos Wavelet Transform with Mexican Hat wavelet'''
    '''https://www.bioconductor.org/packages/devel/bioc/manuals/MassSpecWavelet/man/MassSpecWavelet.pdf'''
    '''https://academic.oup.com/bioinformatics/article/22/17/2059/274284'''
    '''https://pmc.ncbi.nlm.nih.gov/articles/PMC2631518/'''

    cwt_matrix, scales = get_cwt_matrix(signal, scales)
    local_maxima = get_local_maxima_per_scale(cwt_matrix, scales)
    ridge_idx, ridge_list, peak_status = link_ridges(local_maxima, scales)
    peaks = pick_peaks(signal, ridge_list, ridge_idx, cwt_matrix, scales)
    
    peak_mask = np.zeros(signal.size, dtype=np.int64)
    noise_mask = np.ones(signal.size, dtype=np.int64)
    peaks = np.array(peaks, dtype=np.int64)
    for peak in peaks:
        peak_mask[peak] = 1
        noise_mask[peak] = 0
    np.savetxt("results/peak_mask.csv", peak_mask, delimiter=",")
    
    return peak_mask, noise_mask

@njit
def generate_mexican_hat_wavelet(x):
    return (2 / np.sqrt(3)) * (np.pi**(-0.25)) * (1 - x**2) * np.exp(-x**2 / 2)

def next_power_of_base(n, base):
    return base**int(np.ceil(np.log(n) / np.log(base)))

def extend_length(x, add_length):    
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    nR = x.shape[0]
    nR1 = nR + add_length
    
    left, right = 0, add_length
    
    if right > 0:
        x = np.vstack((x, np.tile(x[-1], (add_length, 1))))
    
    if left > 0:
        x = np.vstack((np.tile(x[0], (add_length, 1)), x))
    
    return x.flatten() if x.shape[1] == 1 else x

def extend_n_base(x, n_level=1, base=2):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    nR = x.shape[0]
    if n_level is None:
        nR1 = next_power_of_base(nR, base)
    else:
        nR1 = int(np.ceil(nR / base**n_level) * base**n_level)
    
    if nR != nR1:
        x = extend_length(x, add_length=nR1 - nR)
    
    return x

@njit 
def get_wavelets(signal_length, scales=None, wavelet_xlimit=8, wavelet_length=1024):
    if not scales:
        scales = np.arange(1, 64, 2)
    
    psi_xval = np.linspace(-wavelet_xlimit, wavelet_xlimit, wavelet_length)
    psi = generate_mexican_hat_wavelet(psi_xval)

    len_signal = max(signal_length, next_power_of_base(signal_length, 2))
    
    psi_xval -= psi_xval[0]
    dxval = psi_xval[1]
    xmax = psi_xval[-1]
    
    prepared_wavelets = np.zeros((len(scales), len_signal), dtype=np.complex128)
    len_waves = np.zeros(len(scales), dtype=np.int64)

    for i, scale in enumerate(scales):
        f = np.zeros(len_signal)
        j = 1 + np.floor(np.arange(scale * xmax + 1) / (scale * dxval)).astype(int)
        
        if len(j) == 1:
            j = np.array([1, 1])
        
        len_wave = len(j)
        f[:len_wave] = np.flip(psi[j - 1]) - np.mean(psi[j - 1])  # Adjust indexing
        
        if len(f) > len_signal:
            break
        
        len_waves[i] = len_wave
        prepared_wavelets[i,:] = np.conj(fft(f))

    return prepared_wavelets, scales, len_waves

@njit
def get_cwt_matrix(signal, scales=None):
    old_len = len(signal)
    prepared_wavelets, scales, len_waves = get_wavelets(old_len)

    signal = extend_n_base(signal, n_level=None, base=2)
    signal_fft = fft(signal)
    len_signal = len(signal)
    cwt_matrix = np.full((len(scales), old_len), np.nan)

    for i, scale_i in enumerate(scales):
        # Convolution via FFT
        w_coefs_i = (1 / np.sqrt(scale_i)) * np.real(ifft(signal_fft * prepared_wavelets[i,:])) / len_signal
        len_wave = len_waves[i]
        
        # Shift the coefficients with half wavelet width
        shift_idx = len_signal - len_wave // 2
        w_coefs_i = np.concatenate((w_coefs_i[shift_idx:], w_coefs_i[:shift_idx]))
        cwt_matrix[i,:] = w_coefs_i[:old_len]

    cwt_matrix = cwt_matrix[:len(scales),:]    
    
    return cwt_matrix, scales

@njit
def find_local_maximum(arr, window_size):
    local_max = np.zeros_like(arr, dtype=int)
    half_win = window_size // 2
    
    for i in range(half_win, len(arr) - half_win):
        if arr[i] == np.max(arr[i - half_win:i + half_win + 1]):
            local_max[i] = 1
    
    max_indices = np.where(local_max > 0)[0]
    if len(max_indices) > 1:
        diffs = np.diff(max_indices)
        to_remove = np.where(diffs < window_size)[0]
        for idx in to_remove:
            if arr[max_indices[idx]] <= arr[max_indices[idx + 1]]:
                local_max[max_indices[idx]] = 0
            else:
                local_max[max_indices[idx + 1]] = 0
    
    return local_max

@njit
def get_local_maxima_per_scale(cwt_matrix, scales, min_win_size=5, amp_th=0):
    local_max = np.full(cwt_matrix.shape, np.nan, dtype=int)
    
    for i, scale in enumerate(scales):
        win_size = max(scale * 2 + 1, min_win_size)
        local_max[i,:] = find_local_maximum(cwt_matrix[i,:], win_size)
    
    # Set values below threshold to zero
    threshold_mask = cwt_matrix < amp_th
    local_max[threshold_mask] = 0

    """ np.savetxt("results/local_max.csv", local_max.T, delimiter=",") """
    
    return local_max

@njit
def link_ridges(local_maxima, scales, step=-1, i_final=0, min_win_size=5, gap_th=3):
    i_init = local_maxima.shape[0]
    n_freq = local_maxima.shape[1]
    max_ind_curr = np.where(local_maxima[i_init - 1, :] > 0)[0]
    col_ind = np.arange(i_init + step, i_final - step, step) if local_maxima.shape[0] > 1 else np.array([0])
    max_ridges = len(max_ind_curr) * len(col_ind)

    ridge_idx = np.full(max_ridges, -1, dtype=np.int32)
    ridge_list = np.full((len(col_ind), max_ridges), -1, dtype=np.int32)
    peak_status = np.zeros(max_ridges, dtype=np.int32)
    remove_j = np.full(max_ridges, -1, dtype=np.int32)
    remove_count = 0

    ridge_count = 0
    for idx in max_ind_curr:
        ridge_idx[ridge_count] = idx
        ridge_list[0, ridge_count] = idx
        ridge_count += 1

    for j in range(len(col_ind)):
        col_j = col_ind[j]
        scale_j = scales[col_j]
        win_size_j = max(int(scale_j * 2 + 1), min_win_size)
        sel_peak_j = np.full(ridge_count, -1, dtype=np.int32)

        for k in range(ridge_count):
            ind_k = ridge_idx[k]
            start_k = max(ind_k - win_size_j, 0)
            end_k = min(ind_k + win_size_j, n_freq)
            ind_curr = np.where(local_maxima[col_j, start_k:end_k] > 0)[0] + start_k

            if len(ind_curr) == 0:
                if peak_status[k] > gap_th and scale_j >= 2:
                    remove_j[remove_count] = k
                    remove_count += 1
                    continue
                else:
                    ind_curr = np.array([ind_k])
                    peak_status[k] += 1
            else:
                peak_status[k] = 0
                ind_curr = np.array([ind_curr[np.argmin(np.abs(ind_curr - ind_k))]])

            ridge_list[j, k] = ind_curr[0]
            sel_peak_j[k] = ind_curr[0]

        max_ind_next = np.where(local_maxima[col_j, :] > 0)[0]
        unselected = np.setdiff1d(max_ind_next, sel_peak_j[sel_peak_j != -1])

        for peak in unselected:
            if ridge_count < max_ridges:
                ridge_idx[ridge_count] = peak
                ridge_list[:, ridge_count] = -1
                ridge_list[j, ridge_count] = peak
                peak_status[ridge_count] = 0
                ridge_count += 1

        if remove_count > 0:
            keep_mask = np.ones(ridge_count, dtype=np.bool_)
            keep_mask[remove_j[:remove_count]] = False
            ridge_idx = ridge_idx[keep_mask]
            ridge_list = ridge_list[:, keep_mask]
            peak_status = peak_status[keep_mask]
            ridge_count -= remove_count
            remove_count = 0

        max_ind_curr = np.concatenate((sel_peak_j[sel_peak_j != -1], unselected))
        ###### SHOULD REMOVE ROWS THAT CONTAIN ONLY -1 AT END TOO
    np.savetxt("results/ridge_list.csv", ridge_list.T, delimiter=",")
    return ridge_idx, ridge_list, peak_status

def pick_peaks(signal, ridge_list, ridge_idx, cwt_matrix, scales,
               snr_threshold=1, peak_scale_range=5,
                ridge_length=32, win_size_noise=500, min_noise_level=0.001,
                exclude_boundaries_size=75):
    ###### MAKE THIS JIT COMPATIBLE
    ###### PEAKS DON'T PROPERLY LINE UP WITH THE SIGNAL YET. PERHAPS MAKE USE OF THE SIGNAL VARIABLE
    ##### TO FIND LARGEST PEAK WITHIN <SCALE> POINTS
    n_scales, n_mz = cwt_matrix.shape

    if ridge_length > np.max(scales):
        ridge_length = np.max(scales)

    if np.isscalar(peak_scale_range):
        peakScaleRange_sel = scales[scales >= peak_scale_range]
    else:
        peak_scale_range = np.asarray(peak_scale_range, dtype=float)
        peakScaleRange_sel = scales[(scales >= peak_scale_range[0]) & (scales <= peak_scale_range[1])]
    
    if min_noise_level is None:
        min_noise_level = 0
    elif min_noise_level < 1:
        min_noise_level = np.max(cwt_matrix) * min_noise_level

    n_ridges = ridge_list.shape[1]
    ridgeLen = np.full(n_ridges, ridge_list.shape[0])

    mzInd = np.array([ridge_list[0, i] for i in range(n_ridges)])

    ridge_level = np.zeros(n_ridges, dtype=int)

    order = np.argsort(mzInd)
    mzInd = mzInd[order]
    ridge_list = ridge_list[:, order]
    ridgeLen = ridgeLen[order]
    ridge_level = ridge_level[order]

    peakScale_list = []
    peakCenterInd_list = []
    peakValue_list = []

    for i in range(n_ridges):
        valid_len = int(ridgeLen[i])
        if valid_len <= 0:
            peakScale_list.append(np.nan)
            peakCenterInd_list.append(np.nan)
            peakValue_list.append(0)
            continue
        
        ridge_i = ridge_list[:valid_len,i]
        levels_i = np.arange(ridge_level[i], ridge_level[i] + valid_len)
        scales_i = scales[levels_i]
        
        sel = np.isin(scales_i, peakScaleRange_sel)
        if not np.any(sel):
            peakScale_list.append(scales_i[0])
            peakCenterInd_list.append(ridge_i[0])
            peakValue_list.append(0)
            continue
        
        levels_i = levels_i[sel]
        scales_i = scales_i[sel]
        ridge_i = ridge_i[sel]
        
        effective_scales = scales_i
        effective_ridge = ridge_i
        effective_levels = levels_i
        
        ridge_values = []
        for mz_val, lev in zip(effective_ridge, effective_levels):
            row = int(mz_val)
            col = int(lev)
            ridge_values.append(cwt_matrix[col, row])
        ridge_values = np.array(ridge_values)
        
        if ridge_values.size > 0:
            max_idx = np.argmax(ridge_values)
            peakScale_list.append(effective_scales[max_idx])
            peakCenterInd_list.append(effective_ridge[max_idx])
            peakValue_list.append(ridge_values[max_idx])
        else:
            peakScale_list.append(np.nan)
            peakCenterInd_list.append(np.nan)
            peakValue_list.append(0)
    
    peakValue_arr = np.array(peakValue_list)
    
    col_idx = np.where(scales == 1)[0][0] if np.any(scales == 1) else 0
    noise = np.abs(cwt_matrix[col_idx, :])
    
    peakSNR_list = []
    for k in range(n_ridges):
        ind_k = int(mzInd[k])
        start_k = max(ind_k - win_size_noise, 0)
        end_k = min(ind_k + win_size_noise + 1, n_mz) 
        noise_window = noise[start_k:end_k]
        #print(noise_window, start_k, end_k, noise)
        
        noiseLevel_k = np.quantile(noise_window, 0.95)
        
        if noiseLevel_k < min_noise_level:
            noiseLevel_k = min_noise_level
        snr = peakValue_arr[k] / noiseLevel_k if noiseLevel_k != 0 else np.inf
        peakSNR_list.append(snr)
    peakSNR_arr = np.array(peakSNR_list)
    
    selInd1 = np.array([scales[int(ridgeLen[i]) - 1] >= ridge_length if ridgeLen[i] > 0 else False 
                          for i in range(n_ridges)])
    
    selInd2 = (peakSNR_arr > snr_threshold)
    
    selInd3 = np.array([not (mz < exclude_boundaries_size or mz >= n_mz - exclude_boundaries_size)
                          for mz in mzInd])
    
    selInd = selInd1 & selInd2 & selInd3
    np.savetxt("results/peaks.csv", mzInd[selInd], delimiter=",")
    
    return mzInd[selInd]
