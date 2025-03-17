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

class UnitCost:
    def __init__(self, capex_p, fom, vom, lifetime, discount_rate, capex_e = 0, transformer_capex = 0, length = 0):
        self.capex_p = capex_p
        self.capex_e = capex_e
        self.fom = fom
        self.vom = vom
        self.lifetime = lifetime
        self.discount_rate = discount_rate
        
        self.transformer_capex = transformer_capex
        self.length = length

@njit
def get_present_value(discount_rate, lifetime):
    return (1-(1+discount_rate)**(-1*lifetime))/discount_rate

@njit
def annualisation_component(power_capacity, energy_capacity, annual_generation, capex_p, capex_e, fom, vom, lifetime, discount_rate):
    present_value = get_present_value(discount_rate, lifetime)
    annualised_cost = (energy_capacity * pow(10,6) * capex_e + power_capacity * pow(10,6) * capex_p) / present_value + power_capacity * pow(10,6) * fom + annual_generation * pow(10,3) * vom if present_value > 0 else power_capacity * pow(10,6) * fom + annual_generation * pow(10,3) * vom
    
    #print(capex_p,capex_e,fom,vom,lifetime,discount_rate,annualised_cost,annual_generation,annual_generation * pow(10,3) * vom,power_capacity * pow(10,6) * fom )
    return annualised_cost

@njit   
def annualisation_transmission(power_capacity, annual_energy_flows, capex_p, fom, vom, lifetime, discount_rate, transformer_capex, length):
    present_value = get_present_value(discount_rate, lifetime)

    return (power_capacity * pow(10,3) * length * capex_p + power_capacity * pow(10,3) * transformer_capex) / present_value + power_capacity * pow(10,3) * length * fom + annual_energy_flows * pow(10,3) * vom

@njit
def calculate_costs(solution): 
    generator_capacities = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    generator_annual_generations = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    storage_p_capacities = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    storage_e_capacities = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    storage_annual_discharge = np.zeros(max(solution.storage_ids)+1, dtype=np.float64)
    line_capacities = np.zeros(max(solution.line_ids)+1, dtype=np.float64)
    line_annual_flows = np.zeros(max(solution.line_ids)+1, dtype=np.float64)

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
        generator_capacities[gen_idx] = solution.CPeak[idx]
        generator_annual_generations[gen_idx] = solution.GFlexible_annual[idx]

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
            solution.generator_costs[5,idx]
        ) for idx in range(0,len(generator_capacities))
        if generator_capacities[idx] > 0
        ], dtype=np.float64).sum()
    
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
            solution.storage_costs[5,idx]
        ) for idx in range(0,len(storage_p_capacities))
        if storage_p_capacities[idx] > 0
        ], dtype=np.float64).sum()
    
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
            solution.line_lengths[idx]
        ) for idx in range(0,len(line_capacities))
        if line_capacities[idx] > 0
        ], dtype=np.float64).sum()

    costs = generator_costs + storage_costs + transmission_costs
    #print(costs)
        
    return costs