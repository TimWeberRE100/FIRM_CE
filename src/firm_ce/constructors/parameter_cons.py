import calendar
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import ValidationError
from firm_ce.common.helpers import parse_comma_separated, parse_float_list
from firm_ce.system.parameters import ScenarioParameters, ScenarioParameters_InstanceType


def determine_interval_parameters(
    first_year: int,
    year_count: int,
    resolution: float,
) -> Tuple[int, NDArray, int]:
    """
    Calculate parameters associated with time intervals, accounting for leap years. The first_year
    and last_year in `config/scenarios.csv` determines whether or not an interval is considered
    a leap year

    Parameters:
    -------
    first_year (int): The first year of the scenario, specified in `config/scenarios.csv`.
    year_count (int): The total number of years in the scenario.
    resolution (float): The time resolution of each interval for the input data [hours/interval].

    Returns:
    -------
    Tuple[int, NDArray, int]: A tuple containing the number of leap days in the scenario,
        a numpy array specifying the first time interval of each year, and the total number
        of time intervals in the scenario.
    """
    year_first_t = np.zeros(year_count, dtype=np.int64)

    leap_days = 0
    for i in range(year_count):
        year = first_year + i
        first_t = i * (8760 // resolution)

        leap_days_so_far = calendar.leapdays(first_year, year)

        leap_adjust = leap_days_so_far * (24 // resolution)
        year_first_t[i] = first_t + leap_adjust

        leap_days += calendar.leapdays(year, year + 1)

    hours_total = year_count * 8760 + leap_days * 24
    intervals_count = int(hours_total // resolution)

    return leap_days, year_first_t, intervals_count


def determine_annual_demand_scalars(
    annual_demand_scalar_raw: object,
    year_count: int,
) -> NDArray[np.float64]:
    """
    Parse the annual_demand_scalar field from scenarios.csv into a per-year multiplier array.

    When a single value s is provided, demand for year i (0-indexed) is scaled by s^i, so
    the first year is unscaled (s^0 = 1.0) and each subsequent year compounds by s. When a
    comma-separated list is provided the values are used directly as per-year multipliers;
    the list must contain exactly year_count entries. An absent or NaN field returns an
    all-ones array (no scaling).

    Parameters:
    -------
    annual_demand_scalar_raw (object): Raw value from the `annual_demand_scalar` column of
        `config/scenarios.csv`. May be a float NaN (absent), a numeric string representing a
        single scalar, or a comma-separated string of floats.
    year_count (int): Number of years in the modelling horizon (firstyear to finalyear
        inclusive).

    Returns:
    -------
    NDArray[np.float64]: Array of per-year demand multipliers of length year_count.

    Exceptions:
    -------
    ValidationError: Raised when a list is provided but its length does not equal year_count.
    """
    values: List[float] = parse_float_list(annual_demand_scalar_raw)

    if not values:
        return np.ones(year_count, dtype=np.float64)

    if len(values) == 1:
        s = values[0]
        return np.array([s ** i for i in range(year_count)], dtype=np.float64)

    if len(values) != year_count:
        raise ValidationError(
            f"annual_demand_scalar has {len(values)} entries but scenario has {year_count} years "
            f"(firstyear to finalyear inclusive). Provide either a single scalar or exactly "
            f"{year_count} comma-separated values."
        )
    return np.array(values, dtype=np.float64)


def determine_investment_steps(
    investment_steps_raw: object,
    first_year: int,
    final_year: int,
) -> NDArray:
    """
    Parse the raw investment_steps value from scenarios.csv into a sorted array of years.
    If the field is absent or NaN, every year in the modelling horizon is treated as an
    investment step.

    Parameters:
    -------
    investment_steps_raw (object): The raw value read from the `investment_steps` column of
        `config/scenarios.csv`. May be a float NaN (missing), an empty string, or a
        comma-separated string of integer years.
    first_year (int): The first year of the scenario.
    final_year (int): The final year of the scenario (inclusive).

    Returns:
    -------
    NDArray: A 1-D int64 array of investment step years.
    """
    if isinstance(investment_steps_raw, float) and np.isnan(investment_steps_raw):
        return np.array(range(first_year, final_year + 1), dtype=np.int64)
    return np.array(
        [int(s) for s in parse_comma_separated(str(investment_steps_raw), lower=False)], dtype=np.int64
    )


def build_generation_parameters_for_pathway_step(
    base_static: ScenarioParameters_InstanceType,
    n_weather_years: int,
    intervals_per_weather_year: int,
    year_energy_demand: float,
) -> ScenarioParameters_InstanceType:
    """
    Build a ScenarioParameters instance for evaluating a pathway planning investment step
    over W weather years using a single investment step year's demand level.

    The resulting instance has year_count = n_weather_years and equal-spaced year boundaries
    derived from intervals_per_weather_year. All other fields (resolution, allowance, node
    count, etc.) are inherited from base_static.

    LCOE correctness: both fixed and variable costs scale with year_count = W, and the
    demand denominator also scales by W, so LCOE = (W*fixed + W*var) / (W*demand) is
    equivalent to the correct single-year LCOE.

    Parameters:
    -------
    base_static (ScenarioParameters_InstanceType): The demand-based ScenarioParameters for the
        scenario; supplies resolution, allowance, node_count, and other inherited fields.
    n_weather_years (int): Number of weather years W in the full generation trace.
    intervals_per_weather_year (int): Number of time intervals in one weather year (T_Y),
        equal to the interval count of the investment step year's demand slice.
    year_energy_demand (float): Total annual energy demand [GWh] for the investment step year,
        used to populate year_energy_demand for every weather year in the result.

    Returns:
    -------
    ScenarioParameters_InstanceType: A ScenarioParameters instance with W equal-length weather
        years covering T_gen = W * T_Y total intervals.
    """
    total_gen_intervals = n_weather_years * intervals_per_weather_year
    weather_year_first_t = np.arange(n_weather_years, dtype=np.int64) * intervals_per_weather_year

    gen_params = ScenarioParameters(
        base_static.resolution,
        base_static.allowance,
        base_static.first_year,
        base_static.final_year,
        n_weather_years,
        0,  # leap_year_count: weather years use equal spacing; negligible FOM effect
        weather_year_first_t,
        total_gen_intervals,
        base_static.node_count,
        base_static.investment_steps,
    )
    gen_params.year_energy_demand = np.full(n_weather_years, year_energy_demand, dtype=np.float64)
    return gen_params


def construct_ScenarioParameters_object(
    scenario_data_dict: Dict[str, str],
    node_count: int,
) -> ScenarioParameters_InstanceType:
    """
    Takes data required to initialise the ScenarioParameters object, casts values into Numba-compatible
    types, and returns an instance of the ScenarioParameters jitclass. The ScenarioParameters are static
    data referenced by the unit committment model.

    Parameters:
    -------
    scenario_data_dict (Dict[str, str]): A dictionary containing data for a single scenario,
        imported from `config/scenarios.csv`.
    node_count (int): The number of nodes (buses) in the network for the scenario.

    Returns:
    -------
    ScenarioParameters_InstanceType: A static instance of the ScenarioParameters jitclass.
    """
    resolution = float(scenario_data_dict.get("resolution", 0.0))
    allowance = float(scenario_data_dict.get("allowance", 0.0))
    first_year = int(scenario_data_dict.get("firstyear", 0))
    final_year = int(scenario_data_dict.get("finalyear", 0))
    year_count = final_year - first_year + 1
    leap_year_count, year_first_t, intervals_count = determine_interval_parameters(
        first_year,
        year_count,
        resolution,
    )

    investment_steps = determine_investment_steps(
        scenario_data_dict.get("investment_steps", ""),
        first_year,
        final_year,
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
        investment_steps,
    )
