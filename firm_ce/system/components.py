from typing import Dict
import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.io.file_manager import DataFile 
from firm_ce.system.costs import UnitCost
from firm_ce.system.topology import Line, Node

if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean
    from numba.experimental import jitclass

    fuel_spec = [
        ('id',int64),
        ('name',string),
        ('cost',float64),
        ('emissions',float64),
    ]
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator    
    fuel_spec = []

@jitclass(fuel_spec)
class Fuel:
    """
    Represents a fuel type with associated cost and emissions.
    """

    def __init__(self, id: int, fuel_dict: Dict[str, str]) -> None:
        """
        Initialize a Fuel object.

        Parameters:
        -------
        id (int): Unique identifier for the fuel.
        fuel_dict (Dict[str, str]): Dictionary containing 'name', 'cost', and 'emissions' keys.
        """

        self.id = int(id)
        self.name = str(fuel_dict['name'])
        self.cost = float(fuel_dict['cost']) # $/GJ
        self.emissions = float(fuel_dict['emissions']) # kg/GJ

    def __repr__(self):
        return f"<Fuel object [{self.id}]{self.name}>"

if JIT_ENABLED:
    generator_spec = [
        ('id',int64),
        ('name',string),
        ('node',Node.class_type.instance_type),
        ('fuel',Fuel.class_type.instance_type),
        ('unit_size',float64),
        ('max_build',float64),
        ('min_build',float64),
        ('capacity',float64),
        ('line',Line.class_type.instance_type),
        ('unit_type',string),
        ('near_optimum_check',boolean),
        ('group',string),
        ('cost',UnitCost.class_type.instance_type),
    ]
else:
    generator_spec = []

@jitclass(generator_spec)
class Generator:
    """
    Represents a generator unit within the system.

    Solar, wind and baseload generators require generation trace datafiles. Flexible 
    generators require datafiles for annual generation limits. Datafiles must be stored in
    the 'data' folder and referenced in 'config/datafiles.csv'.
    """

    def __init__(self, id: int, generator_dict: Dict[str, str], fuel: Fuel, line: Line) -> None:
        """
        Initialize a Generator object.

        Parameters:
        -------
        id (int): Unique identifier for the generator.
        generator_dict (Dict[str, str]): Dictionary containing generator attributes.
        fuel (Fuel): The associated fuel object.
        line (Line): The generic minor line defined to connect the generator to the transmission network.
                        Minor lines should have empty node_start and node_end values. They do not form part
                        of the network topology, but are used to estimate connection costs.
        """

        self.id = id
        self.name = str(generator_dict['name'])
        self.node = str(generator_dict['node'])
        self.fuel = fuel
        self.unit_size = float(generator_dict['unit_size']) # GW/unit
        self.max_build = float(generator_dict['max_build'])  # GW/year
        self.min_build = float(generator_dict['min_build'])  # GW/year
        self.capacity = float(generator_dict['initial_capacity'])  # GW
        self.line = line
        self.unit_type = str(generator_dict['unit_type'])
        self.near_optimum_check = str(generator_dict.get('near_optimum','')).lower() in ('true','1','yes')
        
        raw_group = generator_dict.get('range_group', '')
        if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == '':
            self.group = self.name  
        else:
            self.group = str(raw_group).strip()
            
        self.cost = UnitCost(capex_p=float(generator_dict['capex']),
                              fom=float(generator_dict['fom']),
                              vom=float(generator_dict['vom']),
                              lifetime=int(generator_dict['lifetime']),
                              discount_rate=float(generator_dict['discount_rate']),
                              heat_rate_base=float(generator_dict['heat_rate_base']), # GJ/unit-h
                              heat_rate_incr=float(generator_dict['heat_rate_incr']), # GJ/MWh
                              fuel=fuel)
        
        self.generation_trace = None
        self.annual_limit = 0
    
    def load_datafile(self, datafiles: Dict[str, DataFile]) -> None:
        """
        Load generation trace or annual generation limit data for this generator.

        Generation traces represent the interval capacity factor and annual generation 
        limits should have units GWh/year.

        Parameters:
        -------
        datafiles (Dict[str, DataFile]): A dictionary of named DataFile objects.
        """
        for key, datafile in datafiles.items():
            if self.name not in datafile.data:
                continue

            match datafile.type:
                case 'generation':
                    self.generation_trace = np.array(datafile.data[self.name], dtype=np.float64)
                    break
                case 'flexible_annual_limit':
                    self.annual_limit = np.array(datafile.data[self.name], dtype=np.float64)
                    break
                case _:
                    continue    
        return None

    def unload_datafile(self) -> None:     
        """
        Unload any attached data to free memory.
        """  
        self.generation_trace = None
        self.annual_limit = 0     

    def __repr__(self):
        return f"<Generator object [{self.id}]{self.name}>"

if JIT_ENABLED:
    storage_spec = [
        ('id',int64),
        ('name',string),
        ('node',Node.class_type.instance_type),
        ('power_capacity',float64),
        ('energy_capacity',float64),
        ('max_build_p',float64),
        ('max_build_e',float64),
        ('min_build_p',float64),
        ('min_build_e',float64),        
        ('line',Line.class_type.instance_type),
        ('unit_type',string),
        ('near_optimum_check',boolean),
        ('group',string),
        ('cost',UnitCost.class_type.instance_type),
    ]
else:
    storage_spec = []

@jitclass(storage_spec)
class Storage:
    """
    Represents an energy storage system unit in the system.
    """
    def __init__(self, id: int, storage_dict: Dict[str, str], line: Line) -> None:
        """
        Initialize a Storage object.

        Parameters:
        -------
        id (int): Unique identifier for the storage unit.
        storage_dict (Dict[str, str]): Dictionary containing storage attributes.
        line (Line): The generic minor line defined to connect the generator to the transmission network.
                        Minor lines should have empty node_start and node_end values. They do not form part
                        of the network topology, but are used to estimate connection costs.
        """

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
        self.near_optimum_check = str(storage_dict.get('near_optimum','')).lower() in ('true','1','yes')
        
        raw_group = storage_dict.get('range_group', '')
        if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == '':
            self.group = self.name  
        else:
            self.group = str(raw_group).strip()
            
        self.cost = UnitCost(capex_p=float(storage_dict['capex_p']),
                              fom=float(storage_dict['fom']),
                              vom=float(storage_dict['vom']),
                              lifetime=int(storage_dict['lifetime']),
                              discount_rate=float(storage_dict['discount_rate']),
                              capex_e=float(storage_dict['capex_e']),
                              )

    def __repr__(self):
        return f"<Storage object [{self.id}]{self.name}>"

