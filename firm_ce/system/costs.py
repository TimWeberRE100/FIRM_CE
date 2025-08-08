import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba import njit
    from numba.core.types import float64, int64, string, boolean, DictType, UniTuple
    from numba.experimental import jitclass

    unitcost_spec = [
        ('capex_p', float64),
        ('fom', float64),
        ('vom', float64),
        ('lifetime', int64),
        ('discount_rate', float64),
        ('fuel_cost_mwh', float64),
        ('fuel_cost_h', float64),
        ('capex_e', float64),
        ('transformer_capex', float64),
    ]
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper
    
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    
    unitcost_spec = []

@jitclass(unitcost_spec)
class UnitCost:
    """
    Represents cost parameters for a generator, storage, or line object.
    """
    def __init__(self, 
                 capex_p,
                 fom,
                 vom,
                 lifetime,
                 discount_rate,
                 fuel_cost_mwh,
                 fuel_cost_h,
                 capex_e,
                 transformer_capex,
                 ) -> None:
        """
        Initialize cost attributes for a Generator, Storage or Line object.

        Parameters:
        -------
        capex_p (float): Power capacity capital cost ($/kW for generator/storage, $/MW-km for line)
        fom (float): Fixed O&M cost ($/kW/year for generator/storage, $/MW/km/year for line)
        vom (float): Variable O&M cost ($/MWh)
        lifetime (int): Asset lifetime in years
        discount_rate (float): Annual discount rate in range [0,1]
        heat_rate_base (float): Constant heat rate term (GJ/h)
        heat_rate_incr (float): First order marginal heat rate term (GJ/MWh)
        fuel (Fuel): Fuel object
        capex_e (float): Energy capacity capital cost ($/kWh for storage only)
        transformer_capex (float): Transformer-specific cost ($/MW)
        length (float): Line length (used for scaling costs and transmission losses)
        """

        self.capex_p = capex_p # $/kW
        self.capex_e = capex_e # $/kWh, non-zero for energy storage
        self.fom = fom # $/kW/year
        self.vom = vom # $/MWh
        self.lifetime = lifetime # years
        self.discount_rate = discount_rate # [0,1]

        self.fuel_cost_mwh = fuel_cost_mwh # $/MWh = $/GJ * GJ/MWh
        self.fuel_cost_h = fuel_cost_h # $/h/unit = $/GJ * GJ/h/unit
        
        self.transformer_capex = transformer_capex # $/kW, non-zero for lines

UnitCost_InstanceType = UnitCost.class_type.instance_type

@njit
def get_present_value(discount_rate: np.float64, lifetime: np.float64) -> np.float64:
    """
    Calculate the present value of an annuity over a given lifetime.

    Parameters:
    -------
    discount_rate (np.float64): Discount rate (decimal in range [0,1])
    lifetime (np.float64): Number of years

    Returns:
    -------
    np.float64: Present value of a $1/year annuity over 'lifetime' years.
    """
    return (1-(1+discount_rate)**(-1*lifetime))/discount_rate

@njit
def annualisation_component(power_capacity: np.float64, 
                            energy_capacity: np.float64, 
                            annual_generation: np.float64, 
                            capex_p: np.float64, 
                            capex_e: np.float64, 
                            fom: np.float64, 
                            vom: np.float64, 
                            lifetime: np.float64, 
                            discount_rate: np.float64, 
                            fuel_mwh: np.float64, 
                            fuel_h: np.float64, 
                            annual_hours: np.float64,
                            generator_unit_size: np.float64,
                            leap_year_scalar: np.float64,
                            ) -> np.float64:
    """
    Compute the annualised cost of a generator or storage unit.

    Parameters:
    -------
    power_capacity (np.float64): GW
    energy_capacity (np.float64): GWh (0 for generators)
    annual_generation (np.float64): GWh
    capex_p (np.float64): $/kW capital cost
    capex_e (np.float64): $/kWh capital cost (0 for generators)
    fom (np.float64): Fixed O&M ($/kW/year)
    vom (np.float64): Variable O&M ($/MWh)
    lifetime (np.float64): Years
    discount_rate (np.float64): Decimal in range [0,1]
    fuel_mwh (np.float64): Fuel cost per MWh (based upon heat_rate_incr)
    fuel_h (np.float64): Fuel cost per hour (based upon heat_rate_base)
    annual_hours (np.float64): Hours operated annually
    generator_unit_size (np.float64): Capacity of a single unit (GW/unit)
    leap_year_scalar (np.float64): Adjusts average annual FOM to account for leap days in planning period

    Returns:
    -------
    np.float64: Total annualised cost in $
    """

    present_value = get_present_value(discount_rate, lifetime)
    if present_value > 0.001:
        annualised_cost = (
            annualised_build_cost(present_value, power_capacity, energy_capacity, capex_p, capex_e)
            + fom_annual(power_capacity, fom, leap_year_scalar)
            + vom_annual(annual_generation, vom)
            + fuel_annual(annual_generation, power_capacity, generator_unit_size, fuel_mwh, annual_hours, fuel_h)
        )
    else:
        annualised_cost = (
            fom_annual(power_capacity, fom, leap_year_scalar)
            + vom_annual(annual_generation, vom)
            + fuel_annual(annual_generation, power_capacity, generator_unit_size, fuel_mwh, annual_hours, fuel_h)
        )
    return annualised_cost

@njit   
def annualisation_transmission(power_capacity: np.float64, 
                               annual_energy_flows: np.float64, 
                               capex_p: np.float64, 
                               fom: np.float64, 
                               vom: np.float64, 
                               lifetime: np.float64, 
                               discount_rate: np.float64, 
                               transformer_capex: np.float64, 
                               length: np.float64,
                                leap_year_scalar: np.float64,
                               ) -> np.float64:
    """
    Compute annualised cost for a transmission line.

    Parameters:
    -------
    power_capacity (np.float64): GW
    annual_energy_flows (np.float64): GWh
    capex_p (np.float64): $/MW/km
    fom (np.float64): $/MW/km/year
    vom (np.float64): $/MWh
    lifetime (np.float64): Years
    discount_rate (np.float64): Decimal in range [0,1]
    transformer_capex (np.float64): $/MW capital cost of transformers
    length (np.float64): Line length in km
    leap_year_scalar (np.float64): Adjusts average annual FOM to account for leap days in planning period

    Returns:
    -------
    np.float64: Annualised cost in $
    """

    present_value = get_present_value(discount_rate, lifetime)

    return (power_capacity * pow(10,3) * length * capex_p + power_capacity * pow(10,3) * transformer_capex) / present_value + power_capacity * pow(10,3) * length * fom * leap_year_scalar + annual_energy_flows * pow(10,3) * vom 

@njit
def annualised_build_cost(present_value, power_capacity, energy_capacity, capex_p, capex_e):
    return (energy_capacity * 1e6 * capex_e + power_capacity * 1e6 * capex_p) / present_value

@njit
def fom_annual(power_capacity, fom, leap_year_scalar):
    return power_capacity * 1e6 * fom * leap_year_scalar

@njit 
def vom_annual(annual_generation, vom):
    return annual_generation * 1e3 * vom

@njit
def fuel_annual(annual_generation, power_capacity, generator_unit_size, fuel_mwh, annual_hours, fuel_h):
    return annual_generation * 1e3 * fuel_mwh + annual_hours * fuel_h * (power_capacity/generator_unit_size)

@njit
def calculate_costs(solution) -> Tuple[np.float64, 
                                       Tuple[NDArray[np.float64]], 
                                       Tuple[NDArray[np.float64]], 
                                       Tuple[NDArray[np.float64]]]: 
    """
    Compute total annualised system costs for a given solution.

    Parameters:
    -------
    solution (Solution_SingleTime): The solution object associated with a specific x in the differential evolution.

    Returns:
    -------
    Tuple of:
    - total_cost (np.float64): Average annual system cost for the total system ($)
    - tech_costs (Tuple[NDArray[np.float64]]): tuple of arrays (average annual costs for generators, storage, transmission)
    - annual_gen (Tuple[NDArray[np.float64]]): tuple of arrays (average annual generator output, storage discharged energy, line flows)
    - capacities (Tuple[NDArray[np.float64]]): tuple of arrays (generator GW, storage GW, storage GWh, line GW)
    """
    generator_capacities = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    generator_annual_generations = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    generator_annual_hours = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    storage_p_capacities = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    storage_e_capacities = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    storage_annual_discharge = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    line_capacities = np.zeros(max(solution.line_ids)+1, dtype=np.float64)
    line_annual_flows = np.zeros(max(solution.line_ids)+1, dtype=np.float64)
    line_lengths = np.zeros(max(solution.line_ids)+1, dtype=np.float64)

    for idx in range(0,len(solution.pv_cost_ids)):
        gen_idx = solution.pv_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CPV[idx]
        generator_annual_generations[gen_idx] = solution.GPV_annual[idx]

    for idx in range(0,len(solution.wind_cost_ids)):
        gen_idx = solution.wind_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CWind[idx]
        generator_annual_generations[gen_idx] = solution.GWind_annual[idx]

    for idx in range(0,len(solution.flexible_cost_ids)):
        gen_idx = solution.flexible_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CFlexible[idx]
        generator_annual_generations[gen_idx] = solution.GFlexible_annual[idx]
        generator_annual_hours[gen_idx] = solution.Flexible_hours_annual[idx]

    for idx in range(0,len(solution.baseload_cost_ids)):
        gen_idx = solution.baseload_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CBaseload[idx]
        generator_annual_generations[gen_idx] = solution.GBaseload_annual[idx]

    for idx in range(0,len(solution.storage_cost_ids)):
        storage_idx = solution.storage_cost_ids[idx]
        storage_p_capacities[storage_idx] = solution.CPHP[idx]
        storage_e_capacities[storage_idx] = solution.CPHS[idx]
        storage_annual_discharge[storage_idx] = solution.GDischarge_annual[idx]

    for idx in range(0,len(solution.line_cost_ids)):
        line_idx = solution.line_cost_ids[idx]
        line_capacities[line_idx] = solution.CTrans[idx]
        line_annual_flows[line_idx] = solution.TFlowsAbs_annual[idx]
        
    for i in range(len(solution.line_ids)):
        line_idx = solution.line_ids[i]
        line_lengths[line_idx] = solution.line_lengths[i]
        for g in range(len(solution.generator_line_ids)):  
            g_idx = solution.generator_ids[g]      
            g_line = solution.generator_line_ids[g]
            if g_line == line_idx:
                line_capacities[line_idx] += generator_capacities[g_idx]

        for s in range(len(solution.storage_line_ids)):  
            s_idx = solution.storage_ids[s]      
            s_line = solution.storage_line_ids[s]
            if s_line == line_idx:
                line_capacities[line_idx] += storage_p_capacities[s_idx]

    generator_costs = np.array([
        annualisation_component(
            generator_capacities[idx],
            0,
            generator_annual_generations[idx],
            solution.generator_costs[0,idx],
            0,
            solution.generator_costs[2,idx],
            solution.generator_costs[3,idx],
            solution.generator_costs[4,idx],
            solution.generator_costs[5,idx],
            solution.generator_costs[6,idx],
            solution.generator_costs[7,idx],
            generator_annual_hours[idx],
            solution.generator_unit_size[idx],
            solution.fom_scalar,
        ) for idx in range(0,len(generator_capacities))
        if generator_capacities[idx] > 0
        ], dtype=np.float64)
    
    storage_costs = np.array([
        annualisation_component(
            storage_p_capacities[idx],
            storage_e_capacities[idx],
            storage_annual_discharge[idx],
            solution.storage_costs[0,idx],
            solution.storage_costs[1,idx],
            solution.storage_costs[2,idx],
            solution.storage_costs[3,idx],
            solution.storage_costs[4,idx],
            solution.storage_costs[5,idx],
            0,
            0,
            0,
            1,
            solution.fom_scalar,
        ) for idx in range(0,len(storage_p_capacities))
        if storage_p_capacities[idx] > 0
        ], dtype=np.float64)
    
    transmission_costs = np.array([
        annualisation_transmission(
            line_capacities[idx],
            line_annual_flows[idx],
            solution.line_costs[0,idx],
            solution.line_costs[2,idx],
            solution.line_costs[3,idx],
            solution.line_costs[4,idx],
            solution.line_costs[5,idx],
            solution.line_costs[6,idx],
            line_lengths[idx],
            solution.fom_scalar,
        ) for idx in range(0,len(line_capacities))
        if line_capacities[idx] > 0
        ], dtype=np.float64)

    costs = generator_costs.sum() + storage_costs.sum() + transmission_costs.sum()
    tech_costs = (generator_costs, storage_costs, transmission_costs)
    capacities = (generator_capacities, storage_p_capacities, line_capacities, storage_e_capacities)
    annual_gen = (generator_annual_generations, storage_annual_discharge, line_annual_flows)
        
    return costs, tech_costs, annual_gen, capacities

def calculate_cost_components(solution) -> Tuple[np.float64, 
                                       Tuple[NDArray[np.float64]], 
                                       Tuple[NDArray[np.float64]], 
                                       Tuple[NDArray[np.float64]]]:
    
    generator_capacities = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    generator_annual_generations = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    generator_annual_hours = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    storage_p_capacities = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    storage_e_capacities = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    storage_annual_discharge = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    line_capacities = np.zeros(max(solution.line_ids)+1, dtype=np.float64)
    line_annual_flows = np.zeros(max(solution.line_ids)+1, dtype=np.float64)
    line_lengths = np.zeros(max(solution.line_ids)+1, dtype=np.float64)

    for idx in range(0,len(solution.pv_cost_ids)):
        gen_idx = solution.pv_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CPV[idx]
        generator_annual_generations[gen_idx] = solution.GPV_annual[idx]

    for idx in range(0,len(solution.wind_cost_ids)):
        gen_idx = solution.wind_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CWind[idx]
        generator_annual_generations[gen_idx] = solution.GWind_annual[idx]

    for idx in range(0,len(solution.flexible_cost_ids)):
        gen_idx = solution.flexible_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CFlexible[idx]
        generator_annual_generations[gen_idx] = solution.GFlexible_annual[idx]
        generator_annual_hours[gen_idx] = solution.Flexible_hours_annual[idx]

    for idx in range(0,len(solution.baseload_cost_ids)):
        gen_idx = solution.baseload_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CBaseload[idx]
        generator_annual_generations[gen_idx] = solution.GBaseload_annual[idx]

    for idx in range(0,len(solution.storage_cost_ids)):
        storage_idx = solution.storage_cost_ids[idx]
        storage_p_capacities[storage_idx] = solution.CPHP[idx]
        storage_e_capacities[storage_idx] = solution.CPHS[idx]
        storage_annual_discharge[storage_idx] = solution.GDischarge_annual[idx]

    for idx in range(0,len(solution.line_cost_ids)):
        line_idx = solution.line_cost_ids[idx]
        line_capacities[line_idx] = solution.CTrans[idx]
        line_annual_flows[line_idx] = solution.TFlowsAbs_annual[idx]
        
    for i in range(len(solution.line_ids)):
        line_idx = solution.line_ids[i]
        line_lengths[line_idx] = solution.line_lengths[i]
        for g in range(len(solution.generator_line_ids)):  
            g_idx = solution.generator_ids[g]      
            g_line = solution.generator_line_ids[g]
            if g_line == line_idx:
                line_capacities[line_idx] += generator_capacities[g_idx]

        for s in range(len(solution.storage_line_ids)):  
            s_idx = solution.storage_ids[s]      
            s_line = solution.storage_line_ids[s]
            if s_line == line_idx:
                line_capacities[line_idx] += storage_p_capacities[s_idx]

    generator_capex = np.array([
        annualised_build_cost(
            pv,
            generator_capacities[idx],
            0,
            solution.generator_costs[0, idx],
            0
        ) if (pv := get_present_value(solution.generator_costs[5, idx], solution.generator_costs[4, idx])) > 0 else 0
        for idx in range(len(generator_capacities))
        if generator_capacities[idx] > 0
    ], dtype=np.float64)

    storage_capex = np.array([
        annualised_build_cost(
            pv,
            storage_p_capacities[idx],
            storage_e_capacities[idx],
            solution.storage_costs[0, idx],
            solution.storage_costs[1, idx]
        ) if (pv := get_present_value(solution.storage_costs[5, idx], solution.storage_costs[4, idx])) > 0 else 0
        for idx in range(len(storage_p_capacities))
        if storage_p_capacities[idx] > 0
    ], dtype=np.float64)

    generator_fom = np.array([
        fom_annual(                              
            generator_capacities[idx],
            solution.generator_costs[2,idx],
            solution.fom_scalar,
        ) for idx in range(0,len(generator_capacities))
        if generator_capacities[idx] > 0
        ], dtype=np.float64)

    storage_fom = np.array([
        fom_annual(                              
            storage_p_capacities[idx],
            solution.storage_costs[2,idx],
            solution.fom_scalar,
        ) for idx in range(0,len(storage_p_capacities))
        if storage_p_capacities[idx] > 0
        ], dtype=np.float64)

    generator_vom = np.array([
        vom_annual(                              
            generator_annual_generations[idx],
            solution.generator_costs[3,idx],
        ) for idx in range(0,len(generator_capacities))
        if generator_capacities[idx] > 0
        ], dtype=np.float64)

    storage_vom = np.array([
        vom_annual(                              
            storage_annual_discharge[idx],
            solution.storage_costs[3,idx],
        ) for idx in range(0,len(storage_p_capacities))
        if storage_p_capacities[idx] > 0
        ], dtype=np.float64)

    generator_fuel = np.array([
        fuel_annual(                              
            generator_annual_generations[idx],
            generator_capacities[idx],
            solution.generator_unit_size[idx],
            solution.generator_costs[6,idx],
            generator_annual_hours[idx],
            solution.generator_costs[7,idx],
        ) for idx in range(0,len(generator_capacities))
        if generator_capacities[idx] > 0
        ], dtype=np.float64)

    storage_fuel = np.array([
        0.0 for idx in range(0,len(storage_p_capacities))
        if storage_p_capacities[idx] > 0
        ], dtype=np.float64)

    output = solution.years*np.array([
        np.concatenate((generator_capex, storage_capex)),
        np.concatenate((generator_fom, storage_fom)),
        np.concatenate((generator_vom, storage_vom)),
        np.concatenate((generator_fuel, storage_fuel))
    ], dtype=np.float64) / 1000

    return output