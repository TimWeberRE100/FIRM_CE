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
    # NEED TO MATCH GENERATOR ID WITH CAPACITIES
    # CPV IDS, CWIND IDS, CFLEXIBLE IDS, CBASELOAD IDS, STORAGE IDS, CTRANS IDS may not be in same order as the IDs in costs
    
    pv_costs = sum([
        annualisation_component(
            power_capacity=solution.CPV[idx],
            annual_generation=solution.GPV_annual[idx],
            capex_p=costs[idx,0],
            fom=costs[idx,2],
            vom=costs[idx,3],
            lifetime=costs[idx,4],
            discount_rate=costs[idx,5]
        ) for idx in range(0,solution.gencost_idx+1)
        if costs[idx,8] == 0
        ])
    
    wind_costs = sum([
        annualisation_component(
            power_capacity=solution.CWind[idx],
            annual_generation=solution.GWind_annual[idx],
            capex_p=costs[idx,0],
            fom=costs[idx,2],
            vom=costs[idx,3],
            lifetime=costs[idx,4],
            discount_rate=costs[idx,5]
        ) for idx in range(0,solution.gencost_idx+1)
        if costs[idx,8] == 1
        ])
    
    flexible_costs = sum([
        annualisation_component(
            power_capacity=solution.CPeak[idx],
            annual_generation=solution.GFlexible_annual[idx],
            capex_p=costs[idx,0],
            fom=costs[idx,2],
            vom=costs[idx,3],
            lifetime=costs[idx,4],
            discount_rate=costs[idx,5]
        ) for idx in range(0,solution.gencost_idx+1)
        if costs[idx,8] == 2
        ])
    
    baseload_costs = sum([
        annualisation_component(
            power_capacity=solution.CBaseload[idx],
            annual_generation=solution.GFlexible_annual[idx],
            capex_p=costs[idx,0],
            fom=costs[idx,2],
            vom=costs[idx,3],
            lifetime=costs[idx,4],
            discount_rate=costs[idx,5]
        ) for idx in range(0,solution.gencost_idx+1)
        if costs[idx,8] == 3
        ])
    
    storage_costs = sum([
        annualisation_component(
            power_capacity=solution.CPHSP[idx],
            energy_capacity=solution.CPHSE[idx],
            annual_generation=solution.GPHES_annual[idx],
            capex_p=costs[idx,0],
            fom=costs[idx,2],
            vom=costs[idx,3],
            lifetime=costs[idx,4],
            discount_rate=costs[idx,5]
        ) for idx in range(solution.gencost_idx+1,solution.storagecost_idx+1)
        if costs[idx,8] == 4
        ])
    
    transmission_costs = sum([
        annualisation_transmission(
            power_capacity=solution.CTrans[idx],
            annual_energy_flows=solution.GTFlowsAbs_annual[idx],
            capex_p=costs[idx,0],
            fom=costs[idx,2],
            vom=costs[idx,3],
            lifetime=costs[idx,4],
            discount_rate=costs[idx,5],
            transformer_capex=costs[idx,6],
            length=costs[idx,7]
        ) for idx in range(solution.storagecost_idx+1,solution.linecost_idx+1)
        if costs[idx,8] == 6
    ])

    #PV_Wind_transmission_cost = annulization_transmission(S.UnitCosts[8],S.UnitCosts[34],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],sum(S.CPV),0,20) 

    costs = pv_costs + wind_costs + storage_costs + flexible_costs + baseload_costs + transmission_costs
        
    return solution