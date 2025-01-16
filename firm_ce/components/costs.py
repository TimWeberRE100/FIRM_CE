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
    costs = solution.costs
    
    print(len(solution.CWind), len(solution.CPV), len(solution.CPeak), len(solution.CBaseload), len(solution.CPHP), len(solution.CTrans))

    capacities = np.zeros(solution.linecost_idx, dtype=np.float64)
    energy_capacities = np.zeros(solution.linecost_idx, dtype=np.float64)
    annual_generation = np.zeros(solution.linecost_idx, dtype=np.float64)
    for idx in range(0,len(solution.pv_ids)):
        component_id = solution.pv_ids[idx]
        capacities[component_id] = solution.CPV[idx]
        annual_generation[component_id] = solution.GPV_annual[idx]
    print(component_id)
    
    for idx in range(0,len(solution.wind_ids)):
        component_id = solution.wind_ids[idx]
        capacities[component_id] = solution.CWind[idx]
        annual_generation[component_id] = solution.GWind_annual[idx]
    print(component_id)

    for idx in range(0,len(solution.flexible_ids)):
        component_id = solution.flexible_ids[idx]
        capacities[component_id] = solution.CPeak[idx]
        annual_generation[component_id] = solution.GFlexible_annual[idx]
    print(component_id, solution.CBaseload)

    for idx in range(0,len(solution.baseload_ids)):
        component_id = solution.baseload_ids[idx]
        capacities[component_id] = solution.CBaseload[idx]
        annual_generation[component_id] = solution.GBaseload_annual[idx]
    print(component_id)

    for idx in range(0,len(solution.CPHP)):
        component_id = solution.storage_ids[idx] + solution.gencost_idx 
        capacities[component_id] = solution.CPHP[idx]
        energy_capacities[component_id] = solution.CPHS[idx]
        annual_generation[component_id] = solution.GDischarge_annual[idx]
    print(component_id, solution.CTrans)

    for idx in range(0,len(solution.CTrans)):
        component_id = solution.line_ids[idx] + solution.storagecost_idx 
        print(component_id)
        capacities[component_id] = solution.CTrans[idx]
        annual_generation[component_id] = solution.TFlowsAbs_annual[idx]
    print(component_id)
    print(costs.shape)
    pv_costs = sum([
        annualisation_component(
            power_capacity=capacities[idx],
            annual_generation=annual_generation[idx],
            capex_p=costs[0,idx],
            fom=costs[2,idx],
            vom=costs[3,idx],
            lifetime=costs[4,idx],
            discount_rate=costs[5,idx]
        ) for idx in range(0,solution.gencost_idx)
        if costs[8,idx] == 0
        ])
    
    wind_costs = sum([
        annualisation_component(
            power_capacity=capacities[idx],
            annual_generation=annual_generation[idx],
            capex_p=costs[0,idx],
            fom=costs[2,idx],
            vom=costs[3,idx],
            lifetime=costs[4,idx],
            discount_rate=costs[5,idx]
        ) for idx in range(0,solution.gencost_idx)
        if costs[8,idx] == 1
        ])
    
    flexible_costs = sum([
        annualisation_component(
            power_capacity=capacities[idx],
            annual_generation=annual_generation[idx],
            capex_p=costs[0,idx],
            fom=costs[2,idx],
            vom=costs[3,idx],
            lifetime=costs[4,idx],
            discount_rate=costs[5,idx]
        ) for idx in range(0,solution.gencost_idx)
        if costs[8,idx] == 2
        ])
    
    baseload_costs = sum([
        annualisation_component(
            power_capacity=capacities[idx],
            annual_generation=annual_generation[idx],
            capex_p=costs[0,idx],
            fom=costs[2,idx],
            vom=costs[3,idx],
            lifetime=costs[4,idx],
            discount_rate=costs[5,idx]
        ) for idx in range(0,solution.gencost_idx)
        if costs[8,idx] == 3
        ])
    
    storage_costs = sum([
        annualisation_component(
            power_capacity=capacities[idx],
            energy_capacity=energy_capacities[idx],
            annual_generation=annual_generation[idx],
            capex_p=costs[0,idx],
            fom=costs[2,idx],
            vom=costs[3,idx],
            lifetime=costs[4,idx],
            discount_rate=costs[5,idx]
        ) for idx in range(solution.gencost_idx,solution.storagecost_idx)
        if costs[8,idx] == 4
        ])
    
    transmission_costs = sum([
        annualisation_transmission(
            power_capacity=capacities[idx],
            annual_energy_flows=annual_generation[idx],
            capex_p=costs[0,idx],
            fom=costs[2,idx],
            vom=costs[3,idx],
            lifetime=costs[4,idx],
            discount_rate=costs[5,idx],
            transformer_capex=costs[6,idx],
            length=costs[7,idx]
        ) for idx in range(solution.storagecost_idx,solution.linecost_idx)
        if costs[8,idx] == 6
    ])

    #PV_Wind_transmission_cost = annulization_transmission(S.UnitCosts[8],S.UnitCosts[34],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],sum(S.CPV),0,20) 

    costs = pv_costs + wind_costs + storage_costs + flexible_costs + baseload_costs + transmission_costs
        
    return costs