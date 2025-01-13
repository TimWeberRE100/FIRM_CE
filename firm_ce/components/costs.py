class UnitCost:
    def __init__(self, capex, fom, vom, lifetime, discount_rate, transformer_capex = 0, length = 0):
        self.capex = capex
        self.fom = fom
        self.vom = vom
        self.lifetime = lifetime
        self.discount_rate = discount_rate
        
        self.transformer_capex = transformer_capex
        self.length = length

    def _get_present_value(self):
        return (1-(1+self.discount_rate)**(-1*self.lifetime))/self.discount_rate

    def annualisation_component(self, power_capacity, annual_generation):
        present_value = self._get_present_value()

        return power_capacity * pow(10,6) * self.capex / present_value + power_capacity * pow(10,6) * self.fom + annual_generation * self.vom
    
    def annualisation_transmission(self,power_capacity, annual_energy_flows):
        present_value = self._get_present_value()

        return (power_capacity * pow(10,3) * self.length * self.capex + power_capacity * pow(10,3) * self.transformer_capex) / present_value + power_capacity * pow(10,3) * self.length * self.fom + annual_energy_flows * self.vom

class SolutionCost:
    def __init__(self, generators, storages, lines):
        """ fuels = {generator.fuel.name for generator in generators}
        self.tech_costs = {}
        for fuel in fuels:
            self.tech_costs[fuel] = {'generators': [generator.cost for generator in generators if generator.fuel.name == fuel],
                                     'connections': [generator.connection_cost for generator in generators if generator.fuel.name == fuel],
                                     'capacities': [generator.capacity for generator in generators if generator.fuel.name == fuel],
                                     'avg_annual_generations': [generator.avg_annual_generation for generator in generators if generator.fuel.name == fuel],
                                     }

        self.storage_costs = [storage.cost for storage in storages]
        self.hv_line_costs = [line.cost for line in lines]

        self.CPV = sum([generator.capacity for generator in generators if generator.fuel.name == "solar"]) """

        self.cost = 1
    
    """ def calculate_costs(self):
        PV_costs = sum(generator.cost.annualisation_component() for generator in self.tech_costs['solar']['generators'])
        PV_Wind_transmission_cost = annulization_transmission(S.UnitCosts[8],S.UnitCosts[34],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],sum(S.CPV),0,20)
        wind_costs = 0
        
        transmission_costs = PV_Wind_transmission_cost
        for i in range(len(S.CHVDC)):
            if S.hvdc_mask[i]: # HVDC line costs
                transmission_costs += annulization_transmission(S.UnitCosts[24],0,S.UnitCosts[25],S.UnitCosts[26],S.UnitCosts[27],S.UnitCosts[-1],S.CHVDC[i],S.TDCabs.sum(axis=0)[i]/S.years,S.DCdistance[i])
            else: # HVAC line + transformer costs
                transmission_costs += annulization_transmission(S.UnitCosts[8],S.UnitCosts[34],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],S.CHVDC[i],S.TDCabs.sum(axis=0)[i]/S.years,S.DCdistance[i])
        
        # Converter and substation costs, a pair of stations per line
        for i in range(len(S.CHVDC)):
            if S.hvdc_mask[i]:
                converter_costs = 2 * annulization(S.UnitCosts[28],S.UnitCosts[29],S.UnitCosts[30],S.UnitCosts[31],S.UnitCosts[-1],sum(S.CHVDC),0)
                transmission_costs += converter_costs

        pv_phes = (1-(1+S.UnitCosts[-1])**(-1*S.UnitCosts[18]))/S.UnitCosts[-1]
        phes_costs = (S.UnitCosts[12] * S.CPHP.sum() * pow(10,6) + S.UnitCosts[13] * S.CPHS.sum() * pow(10,6)) / pv_phes \
                        + S.UnitCosts[14] * S.CPHP.sum() * pow(10,6) + S.UnitCosts[15] * GDischarge.sum() / S.years \
                        + S.UnitCosts[16] * ((1+S.UnitCosts[-1])**(-1*S.UnitCosts[17]) + (1+S.UnitCosts[-1])**(-1*S.UnitCosts[17]*2)) / pv_phes
                            
        pv_battery = (1-(1+S.UnitCosts[-1])**(-1*S.UnitCosts[22]))/S.UnitCosts[-1] # 19, 20, 21, 22
        battery_costs = (S.UnitCosts[19] * S.CBP.sum() * pow(10,6) + S.UnitCosts[20] * S.CBS.sum() * pow(10,6)) / pv_battery \
                        + S.UnitCosts[21] * S.CBS.sum() * pow(10,6)
                                    
        hydro_costs = S.UnitCosts[23] * GHydro
        import_costs = S.UnitCosts[32] * GImports
        baseload_costs = S.UnitCosts[33] * GBaseload

        costs = PV_costs + wind_costs + transmission_costs + phes_costs + battery_costs + hydro_costs + import_costs + baseload_costs
        
        return costs """