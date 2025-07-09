import numpy as np
from numpy.typing import NDArray
from typing import List

from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba import njit    
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper
    
def parse_comma_separated(value: str) -> List[str]:
    """
    Parse a comma-separated string into a list of trimmed, non-empty strings.

    Parameters:
    -------
    value (str): A string containing comma-separated values.

    Returns:
    -------
    List[str]: A list of cleaned strings with whitespace removed and empty entries excluded.
    """
    return [item.strip() for item in value.split(',') if item.strip()]
    
@njit
def set_difference_int(array1: NDArray[np.int64], array2: NDArray[np.int64]) -> NDArray[np.bool_]:
    """
    Compute the set difference of two 1D integer arrays (elements in array1 not in array2),
    in a Numba JIT-compatible way.

    Parameters:
    -------
    array1 (np.ndarray): First input array of integers.
    array2 (np.ndarray): Second input array of integers.

    Returns:
    -------
    np.ndarray: Array of elements in array1 but not in array2.
    """
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
def isin_numba(arr: NDArray[np.number], values: NDArray[np.number]) -> NDArray[np.bool_]:
    """
    Determine whether each element of an array is in a set of values,
    in a Numba JIT-compatible way.

    Parameters:
    -------
    arr (np.ndarray): Array of values to check.
    values (np.ndarray): Array of values to check against.

    Returns:
    -------
    np.ndarray: Boolean array indicating whether each element of arr is in values.
    """
    values_set = set(values)  
    result = np.zeros(arr.shape, dtype=np.bool_)

    for i in range(arr.shape[0]):
        if arr[i] in values_set:
            result[i] = True

    return result

@njit
def sum_positive_values(arr: NDArray[np.number]) -> NDArray[np.number]:
    """
    Sum only the positive values in each column of a 2D array.

    Parameters:
    -------
    arr (np.ndarray): 2D array of numeric values.

    Returns:
    -------
    np.ndarray: 1D array where each element is the sum of positive values in the corresponding column.
    """
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
def scalar_clamp(value: np.float64, lower_bound: np.float64, upper_bound: np.float64) -> float:
    """
    Clamp a scalar value between a lower and upper bound.

    Parameters:
    -------
    value (np.float64): Input value.
    lower_bound (np.float64): Minimum allowed value.
    upper_bound (np.float64): Maximum allowed value.

    Returns:
    -------
    float: Clamped value.
    """
    return max(min(value, upper_bound), lower_bound)