from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray

from firm_ce.system.parameters import ScenarioParameters, ScenarioParameters_InstanceType


def determine_interval_parameters(
        first_year: int,
        year_count: int,
        resolution: float,
) -> Tuple[int, NDArray, int]:
    year_first_t = np.zeros(year_count, dtype=np.int64)

    leap_days = 0
    for i in range(year_count):
        year = first_year + i
        first_t = i * (8760 // resolution)

        leap_days_so_far = sum(
            1 for y in range(first_year, year)
            if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        )

        leap_adjust = leap_days_so_far * (24 // resolution)
        year_first_t[i] = first_t + leap_adjust

        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            leap_days += 1

    hours_total = year_count * 8760 + leap_days * 24
    intervals_count = int(hours_total // resolution)

    return leap_days, year_first_t, intervals_count


def construct_ScenarioParameters_object(
        scenario_data_dict: Dict[str, str],
        node_count: int,
) -> ScenarioParameters_InstanceType:
    resolution = float(scenario_data_dict.get('resolution', 0.0))
    allowance = float(scenario_data_dict.get('allowance', 0.0))
    first_year = int(scenario_data_dict.get('firstyear', 0))
    final_year = int(scenario_data_dict.get('finalyear', 0))
    year_count = final_year - first_year + 1
    leap_year_count, year_first_t, intervals_count = determine_interval_parameters(
        first_year,
        year_count,
        resolution,
    )

    return ScenarioParameters(
        resolution,
        allowance,
        first_year,
        final_year,
        year_count,
        leap_year_count,
        year_first_t,
        intervals_count,
        node_count,
    )
