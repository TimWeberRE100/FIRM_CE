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
def set_difference_int(array1, array2):
    # Replaces np.setdiff1d for JIT compatibility
    count = 0
    for i in range(array1.shape[0]):
        candidate = array1[i]
        found = False
        for j in range(array2.shape[0]):
            if candidate == array2[j]:
                found = True
                break
        if not found:
            count += 1

    result = np.empty(count, dtype=np.int32)
    index = 0
    for i in range(array1.shape[0]):
        candidate = array1[i]
        found = False
        for j in range(array2.shape[0]):
            if candidate == array2[j]:
                found = True
                break
        if not found:
            result[index] = candidate
            index += 1
    return result

@njit
def isin_numba(arr, values):
    # Replaces np.isin for JIT compatibility
    values_set = set(values)  
    result = np.zeros(arr.shape, dtype=np.bool_)

    for i in range(arr.shape[0]):
        if arr[i] in values_set:
            result[i] = True

    return result

@njit
def quantile_95(arr):
    # Replaces np.quantile for JIT compatibility
    n = arr.shape[0]
    if n == 0:
        return 0.0
    temp = arr.copy()
    temp.sort()
    
    pos = 0.95 * (n - 1)
    lower_idx = int(pos)
    upper_idx = lower_idx + 1
    if upper_idx >= n:
        return temp[lower_idx]
    weight = pos - lower_idx
    return temp[lower_idx] * (1 - weight) + temp[upper_idx] * weight

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
def sum_along_axis_n(arr, axis_n):
    rows, cols = arr.shape
    sum_vals = np.zeros(cols, dtype=arr.dtype)
    for j in range(cols):
        sum_vals[j] = arr[axis_n, j]
        for i in range(rows):
            if i != axis_n:
                sum_vals[j] += arr[i, j]
    return sum_vals