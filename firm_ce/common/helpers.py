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

def safe_divide(num: float, denom: float):
    return num / denom if denom != 0 else 0