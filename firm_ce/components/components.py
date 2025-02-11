from typing import Dict
from firm_ce.file_manager import DataFile 
from firm_ce.components.costs import UnitCost

class Generator:
    def __init__(self, id: int, generator_dict: Dict[str, str], datafiles: Dict[str, DataFile]) -> None:
        self.id = id
        self.name = str(generator_dict['name'])
        self.node = str(generator_dict['node'])
        self.fuel = str(generator_dict['fuel']) # Fuel type ### CHANGE TO Fuel OBJECT
        self.max_build = int(generator_dict['max_build'])  # MW/year
        self.min_build = int(generator_dict['min_build'])  # MW/year
        self.capacity = float(generator_dict['initial_capacity'])  # MW
        self.unit_type = str(generator_dict['unit_type'])
        self.cost = UnitCost(int(generator_dict['capex']),
                              float(generator_dict['fom']),
                              float(generator_dict['vom']),
                              int(generator_dict['lifetime']),
                              float(generator_dict['discount_rate']),
                              float(generator_dict['lcoe']))

        self.data = None
        for key in datafiles:
            if datafiles[key].type != 'generation':
                continue
            if self.name not in datafiles[key].data.keys():
                continue
            self.data = list(datafiles[key].data[self.name])
            break

    def __repr__(self):
        return f"<Generator object [{self.id}]{self.name}>"

class Storage:
    def __init__(self, id: int, storage_dict: Dict[str, str]) -> None:
        self.id = id
        self.name = str(storage_dict['name'])
        self.node = str(storage_dict['node'])
        self.power_capacity = float(storage_dict['initial_power_capacity'])  # MW
        self.energy_capacity = int(storage_dict['initial_energy_capacity'])  # MWh
        self.duration = int(storage_dict['duration']) if int(storage_dict['duration']) > 0 else 0
        self.charge_efficiency = float(storage_dict['charge_efficiency'])  # %
        self.discharge_efficiency = float(storage_dict['discharge_efficiency'])  # %
        self.max_build_p = int(storage_dict['max_build_p'])  # MW/year
        self.max_build_e = int(storage_dict['max_build_e'])  # MWh/year
        self.min_build_p = int(storage_dict['min_build_p'])  # MW/year
        self.min_build_e = int(storage_dict['min_build_e'])  # MWh/year
        self.unit_type = str(storage_dict['unit_type'])

        self.cost = UnitCost(int(storage_dict['capex_p']),
                              float(storage_dict['fom']),
                              float(storage_dict['vom']),
                              int(storage_dict['lifetime']),
                              float(storage_dict['discount_rate']),
                              capex_e=int(storage_dict['capex_e']),
                              )

    def __repr__(self):
        return f"<Storage object [{self.id}]{self.name}>"

class Fuel:
    def __init__(self, id: int, name: str) -> None:
        self.id = int(id)
        self.name = str(name)

    def __repr__(self):
        return f"<Fuel object [{self.id}]{self.name}>"