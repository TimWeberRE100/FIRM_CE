import numpy as np

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

def get_present_value(discount_rate, lifetime):
    return (1-(1+discount_rate)**(-1*lifetime))/discount_rate

def annualisation_component(power_capacity, annual_generation, capex_p, fom, vom, lifetime, discount_rate, energy_capacity=0,capex_e=0):
    present_value = get_present_value(discount_rate, lifetime)

    return (energy_capacity * pow(10,6) * capex_e + power_capacity * pow(10,6) * capex_p) / present_value + power_capacity * pow(10,6) * fom + annual_generation * vom
    
def annualisation_transmission(power_capacity, annual_energy_flows, capex_p, fom, vom, lifetime, discount_rate, transformer_capex, length):
    present_value = get_present_value(discount_rate, lifetime)

    return (power_capacity * pow(10,3) * length * capex_p + power_capacity * pow(10,3) * transformer_capex) / present_value + power_capacity * pow(10,3) * length * fom + annual_energy_flows * vom

def calculate_costs(solution): 
    pv_cost_ids = solution.generator_ids[np.where(solution.generator_unit_types == solution.unit_types['solar'])]
    wind_cost_ids = solution.generator_ids[np.where(solution.generator_unit_types == solution.unit_types['wind'])]
    flexible_cost_ids = solution.generator_ids[np.where(solution.generator_unit_types == solution.unit_types['flexible'])]
    baseload_cost_ids = solution.generator_ids[np.where(solution.generator_unit_types == solution.unit_types['baseload'])]

    generator_capacities = np.zeros(len(solution.generator_ids), dtype=np.float64)
    generator_annual_generations = np.zeros(len(solution.generator_ids), dtype=np.float64)

    for idx in range(0,len(pv_cost_ids)):
        gen_idx = pv_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CPV[idx]
        generator_annual_generations[gen_idx] = solution.GPV_annual[idx]

    for idx in range(0,len(wind_cost_ids)):
        gen_idx = wind_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CWind[idx]
        generator_annual_generations[gen_idx] = solution.GWind_annual[idx]

    for idx in range(0,len(flexible_cost_ids)):
        gen_idx = flexible_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CPeak[idx]
        generator_annual_generations[gen_idx] = solution.GFlexible_annual[idx]

    for idx in range(0,len(baseload_cost_ids)):
        gen_idx = baseload_cost_ids[idx]
        generator_capacities[gen_idx] = solution.CBaseload[idx]
        generator_annual_generations[gen_idx] = solution.GBaseload_annual[idx]

    print(len(solution.CWind), len(solution.CPV), len(solution.CPeak), len(solution.CBaseload), len(solution.CPHP), len(solution.CTrans))
    print(generator_capacities)
    print(generator_annual_generations)

    generator_costs = np.array([
        annualisation_component(
            power_capacity=generator_capacities[idx],
            annual_generation=generator_annual_generations[idx],
            capex_p=solution.generator_costs[0,idx],
            fom=solution.generator_costs[2,idx],
            vom=solution.generator_costs[3,idx],
            lifetime=solution.generator_costs[4,idx],
            discount_rate=solution.generator_costs[5,idx]
        ) for idx in range(0,len(generator_capacities))
        ], dtype=np.float64).sum()
    
    storage_costs = np.array([
        annualisation_component(
            power_capacity=solution.CPHP[idx],
            energy_capacity=solution.CPHS[idx],
            annual_generation=solution.Discharge[idx],
            capex_p=solution.storage_costs[0,idx],
            fom=solution.storage_costs[2,idx],
            vom=solution.storage_costs[3,idx],
            lifetime=solution.storage_costs[4,idx],
            discount_rate=solution.storage_costs[5,idx]
        ) for idx in range(0,len(solution.CPHP))
        ], dtype=np.float64).sum()
    
    transmission_costs = np.array([
        annualisation_transmission(
            power_capacity=solution.CTrans[idx],
            annual_energy_flows=solution.line_annual_TFlowsAbs[idx],
            capex_p=solution.line_costs[0,idx],
            fom=solution.line_costs[2,idx],
            vom=solution.line_costs[3,idx],
            lifetime=solution.line_costs[4,idx],
            discount_rate=solution.line_costs[5,idx],
            transformer_capex=solution.line_costs[6,idx],
            length=solution.line_lengths[idx]
        ) for idx in range(0,len(solution.CTrans))
        ], dtype=np.float64).sum()

    #PV_Wind_transmission_cost = annulization_transmission(S.UnitCosts[8],S.UnitCosts[34],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],sum(S.CPV),0,20) 

    costs = generator_costs + storage_costs + transmission_costs
        
    return costs