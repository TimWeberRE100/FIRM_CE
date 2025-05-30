from typing import Dict
from firm_ce.io.file_manager import DataFile 
from firm_ce.system.costs import UnitCost
from firm_ce.system.topology import Line

class Fuel:
    def __init__(self, id: int, fuel_dict: Dict[str, str]) -> None:
        self.id = int(id)
        self.name = str(fuel_dict['name'])
        self.cost = float(fuel_dict['cost'])
        self.emissions = float(fuel_dict['emissions'])

    def __repr__(self):
        return f"<Fuel object [{self.id}]{self.name}>"

class Generator:
    def __init__(self, id: int, generator_dict: Dict[str, str], fuel: Fuel, line: Line) -> None:
        self.id = id
        self.name = str(generator_dict['name'])
        self.node = str(generator_dict['node'])
        self.fuel = fuel
        self.max_build = float(generator_dict['max_build'])  # GW/year
        self.min_build = float(generator_dict['min_build'])  # GW/year
        self.capacity = float(generator_dict['initial_capacity'])  # GW
        self.line = line
        self.unit_type = str(generator_dict['unit_type'])
        self.cost = UnitCost(capex_p=float(generator_dict['capex']),
                              fom=float(generator_dict['fom']),
                              vom=float(generator_dict['vom']),
                              lifetime=int(generator_dict['lifetime']),
                              discount_rate=float(generator_dict['discount_rate']),
                              heat_rate_base=float(generator_dict['heat_rate_base']),
                              heat_rate_incr=float(generator_dict['heat_rate_incr']),
                              fuel=fuel)
        
        self.data = None
        self.annual_limit = 0
    
    def load_datafile(self, datafiles: Dict[str, DataFile]) -> None:
        for key in datafiles:
            if (datafiles[key].type != 'generation') and (datafiles[key].type != 'flexible_annual_limit'):
                continue
            if self.name not in datafiles[key].data.keys():
                continue
            if datafiles[key].type == 'generation':
                self.data = list(datafiles[key].data[self.name])
                break
            elif datafiles[key].type == 'flexible_annual_limit':
                self.annual_limit = list(datafiles[key].data[self.name])
                break    

    def unload_datafile(self) -> None:       
        self.data = None
        self.annual_limit = 0     

    def __repr__(self):
        return f"<Generator object [{self.id}]{self.name}>"

class Storage:
    def __init__(self, id: int, storage_dict: Dict[str, str], line: Line) -> None:
        self.id = id
        self.name = str(storage_dict['name'])
        self.node = str(storage_dict['node'])
        self.power_capacity = float(storage_dict['initial_power_capacity'])  # GW
        self.energy_capacity = float(storage_dict['initial_energy_capacity'])  # GWh
        self.duration = int(storage_dict['duration']) if int(storage_dict['duration']) > 0 else 0
        self.charge_efficiency = float(storage_dict['charge_efficiency'])  # %
        self.discharge_efficiency = float(storage_dict['discharge_efficiency'])  # %
        self.max_build_p = float(storage_dict['max_build_p'])  # GW/year
        self.max_build_e = float(storage_dict['max_build_e'])  # GWh/year
        self.min_build_p = float(storage_dict['min_build_p'])  # GW/year
        self.min_build_e = float(storage_dict['min_build_e'])  # GWh/year
        self.line = line
        self.unit_type = str(storage_dict['unit_type'])

        self.cost = UnitCost(capex_p=float(storage_dict['capex_p']),
                              fom=float(storage_dict['fom']),
                              vom=float(storage_dict['vom']),
                              lifetime=int(storage_dict['lifetime']),
                              discount_rate=float(storage_dict['discount_rate']),
                              capex_e=float(storage_dict['capex_e']),
                              )

    def __repr__(self):
        return f"<Storage object [{self.id}]{self.name}>"

