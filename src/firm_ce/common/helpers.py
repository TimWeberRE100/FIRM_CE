from typing import List
import numpy as np


def parse_comma_separated(value: str, lower: bool = True) -> List[str]:
    """
    Parse a comma-separated string into a list of trimmed, non-empty strings.

    Parameters:
    -------
    value (str): A string containing comma-separated values.
    lower (bool): If True, converts each item to lowercase before returning. Defaults to True.

    Returns:
    -------
    List[str]: A list of strings with leading/trailing whitespace removed and empty entries excluded.
    """
    if lower:
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_divide(num: float, denom: float) -> float:
    """Safe division for calculating levelised costs when total dispatch energy from the asset is 0."""
    return num / denom if denom != 0 else 0.0


def parse_id_list(value: str | float) -> List[int]:
    """
    Parse a comma-separated value into a list of integers, returning [] for NaN or empty.

    Parameters:
    -------
    value (str | float): A comma-separated string, or a NaN float (empty CSV cell).

    Returns:
    -------
    List[int]: Parsed integer list, or empty list for NaN/empty input.
    """
    if isinstance(value, float) and np.isnan(value):
        return []
    return [int(x) for x in parse_comma_separated(str(value), lower=False) if x]


def parse_float_list(value: str | float) -> List[float]:
    """
    Parse a comma-separated value into a list of floats, returning [] for NaN or empty.

    Parameters:
    -------
    value (str | float): A comma-separated string, or a NaN float (empty CSV cell).

    Returns:
    -------
    List[float]: Parsed float list, or empty list for NaN/empty input.
    """
    if isinstance(value, float) and np.isnan(value):
        return []
    return [float(x) for x in parse_comma_separated(str(value), lower=False) if x]