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
    def __init__(self, capex_p, fom, vom, lifetime, discount_rate, capex_e = 0, transformer_capex = 0, length = 0, lcoe = 0):
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
def annualisation_component(power_capacity, annual_generation, capex_p, fom, vom, lifetime, discount_rate, energy_capacity=0,capex_e=0):
    present_value = get_present_value(discount_rate, lifetime)
    annualised_cost = (energy_capacity * pow(10,6) * capex_e + power_capacity * pow(10,6) * capex_p) / present_value + power_capacity * pow(10,6) * fom + annual_generation * vom if present_value > 0 else power_capacity * pow(10,6) * fom + annual_generation * vom
    """ print(f"{power_capacity} || {energy_capacity} || {capex_p} || {capex_e} || {power_capacity * pow(10,6) * fom} || {(energy_capacity * pow(10,6) * capex_e + power_capacity * pow(10,6) * capex_p) / present_value} == {annualised_cost}") """
    return annualised_cost

@njit   
def annualisation_transmission(power_capacity, annual_energy_flows, capex_p, fom, vom, lifetime, discount_rate, transformer_capex, length):
    present_value = get_present_value(discount_rate, lifetime)

    return (power_capacity * pow(10,3) * length * capex_p + power_capacity * pow(10,3) * transformer_capex) / present_value + power_capacity * pow(10,3) * length * fom + annual_energy_flows * vom

@njit
def calculate_costs(solution): 
    generator_capacities = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    generator_annual_generations = np.zeros(max(solution.generator_ids)+1, dtype=np.float64)
    line_capacities = np.zeros(max(solution.line_ids)+1, dtype=np.float64)
    line_annual_flows = np.zeros(max(solution.line_ids)+1, dtype=np.float64)
    #print("Annuals: ", solution.GPV_annual, solution.GWind_annual, solution.GFlexible_annual, solution.GBaseload_annual)

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

    for idx in range(0,len(solution.line_cost_ids)):
        line_idx = solution.line_cost_ids[idx]
        line_capacities[line_idx] = solution.CTrans[idx]
        line_annual_flows[line_idx] = solution.TFlowsAbs_annual[idx]

    #print(len(solution.CWind), len(solution.CPV), len(solution.CPeak), len(solution.CBaseload), len(solution.CPHP), len(solution.CTrans))
    #print(generator_capacities)
    #print(generator_annual_generations)

    generator_newbuild_costs = np.array([
        annualisation_component(
            power_capacity=generator_capacities[idx],
            annual_generation=generator_annual_generations[idx],
            capex_p=solution.generator_costs[0,idx],
            fom=solution.generator_costs[2,idx],
            vom=solution.generator_costs[3,idx],
            lifetime=solution.generator_costs[4,idx],
            discount_rate=solution.generator_costs[5,idx]
        ) for idx in range(0,len(generator_capacities))
        if generator_capacities[idx] > 0
        if idx not in solution.flexible_ids
        ], dtype=np.float64).sum()

    """ generator_existing_costs = np.array([
        solution.generator_costs[7,idx] * generator_annual_generations[idx] for idx in range(0,len(generator_capacities))
    ], dtype=np.float64).sum() """
    
    transmission_costs = np.array([
        annualisation_transmission(
            power_capacity=line_capacities[idx],
            annual_energy_flows=line_annual_flows[idx],
            capex_p=solution.line_costs[0,idx],
            fom=solution.line_costs[2,idx],
            vom=solution.line_costs[3,idx],
            lifetime=solution.line_costs[4,idx],
            discount_rate=solution.line_costs[5,idx],
            transformer_capex=solution.line_costs[6,idx],
            length=solution.line_lengths[idx]
        ) for idx in range(0,len(line_capacities))
        if line_capacities[idx] > 0
        ], dtype=np.float64).sum()

    #PV_Wind_transmission_cost = annulization_transmission(S.UnitCosts[8],S.UnitCosts[34],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],sum(S.CPV),0,20) 
    #print(generator_newbuild_costs, generator_existing_costs, storage_costs, transmission_costs)

    """ np.savetxt("results/generator_costs.csv", solution.generator_costs, delimiter=",")
    np.savetxt("results/generator_cost_capacities.csv", generator_capacities, delimiter=",") """
    """ np.savetxt("results/storage_costs.csv", solution.storage_costs, delimiter=",")
    np.savetxt("results/generator_cost_capacities.csv", generator_capacities, delimiter=",") """
    costs = generator_newbuild_costs + transmission_costs #+ generator_existing_costs
        
    return costs