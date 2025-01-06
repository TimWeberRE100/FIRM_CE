from typing import Dict

class Generator:
    def __init__(self, id: int, generator_dict: Dict[str, str]) -> None:
        self.id = id
        self.name = str(generator_dict['name'])
        self.node = str(generator_dict['node'])
        self.fuel = str(generator_dict['fuel']) # Fuel type ### CHANGE TO Fuel OBJECT
        self.capex = int(generator_dict['capex'])  # $
        self.fom = float(generator_dict['fom'])  # $/kW-year
        self.vom = float(generator_dict['vom'])  # $/kWh
        self.lifetime = int(generator_dict['lifetime'])  # years
        self.discount_rate = float(generator_dict['discount_rate']) # [0,1]
        self.max_build = int(generator_dict['max_build'])  # MW/year
        self.min_build = int(generator_dict['min_build'])  # MW/year
        self.capacity = int(generator_dict['capacity'])  # MW

    def __repr__(self):
        return f"{self.id}: {self.name}"

class Storage:
    def __init__(self, id: int, storage_dict: Dict[str, str]) -> None:
        self.id = id
        self.name = str(storage_dict['name'])
        self.node = str(storage_dict['node'])
        self.power_capacity = int(storage_dict['power_capacity'])  # MW
        self.energy_capacity = int(storage_dict['energy_capacity'])  # MWh
        self.capex = int(storage_dict['capex'])  # $
        self.fom = float(storage_dict['fom'])  # $/kW-year
        self.vom = float(storage_dict['vom'])  # $/kWh
        self.lifetime = int(storage_dict['lifetime'])  # Years
        self.discount_rate = float(storage_dict['discount_rate']) #[0,1]
        self.charge_efficiency = float(storage_dict['charge_efficiency'])  # %
        self.discharge_efficiency = float(storage_dict['discharge_efficiency'])  # %
        self.max_build_p = int(storage_dict['max_build_p'])  # MW/year
        self.max_build_e = int(storage_dict['max_build_e'])  # MWh/year
        self.min_build_p = int(storage_dict['min_build_p'])  # MW/year
        self.min_build_e = int(storage_dict['min_build_e'])  # MWh/year

    def __repr__(self):
        return f"{self.id}: {self.name}"

class Fuel:
    def __init__(self, id: int, name: str) -> None:
        self.id = int(id)
        self.name = str(name)

    def __repr__(self):
        return f"{self.id}: {self.name}"