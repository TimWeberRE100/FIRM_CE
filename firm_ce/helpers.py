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
def isin_numba(arr, values):
    values_set = set(values)  
    result = np.zeros(arr.shape, dtype=np.bool_)

    for i in range(arr.shape[0]):
        if arr[i] in values_set:
            result[i] = True

    return result

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