import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.system.costs import UnitCost
from firm_ce.system.topology import Line, Node
from firm_ce.system.traces import Traces2d

if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType
    from numba.experimental import jitclass

    fuel_spec = [
        ('static_instance',boolean),
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

    def __init__(self, static_instance, idx, name, cost, emissions) -> None:
        """
        Initialize a Fuel object.

        Parameters:
        -------
        id (int): Unique identifier for the fuel.
        fuel_dict (Dict[str, str]): Dictionary containing 'name', 'cost', and 'emissions' keys.
        """
        
        self.static_instance = static_instance
        self.id = idx
        self.name = name
        self.cost = cost # $/GJ
        self.emissions = emissions # kg/GJ

if JIT_ENABLED:
    generator_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order', int64),
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
        ('data_status',string),
        ('data',float64[:]),
        ('annual_constraints_data',float64[:]),
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

    def __init__(self, 
                 static_instance,
                 idx, 
                 order,
                 name,
                 unit_size,
                 max_build,
                 min_build,
                 capacity,
                 unit_type,
                 near_optimum_check,
                 node,
                 fuel, 
                 line,
                 group,
                 cost,
                 ) -> None:
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
        self.static_instance = static_instance
        self.id = idx
        self.order = order # id specific to scenario
        self.name = name
        self.unit_size = unit_size # GW/unit
        self.max_build = max_build  # GW/year
        self.min_build = min_build  # GW/year
        self.capacity = capacity  # GW        
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check        
        self.node = node
        self.fuel = fuel
        self.line = line
        self.group = group            
        self.cost = cost

        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)
        self.annual_constraints_data = np.empty((0,), dtype=np.float64)

    def build_capacity(self, new_build_power_capacity):
        if self.static_instance:
            raise_static_modification_error()     
        self.capacity += new_build_power_capacity            
        return None
    
    def load_data(self, generation_trace, annual_constraints):
        self.data_status= "availability"
        self.data = generation_trace
        self.annual_constraints_data = annual_constraints

        return None
    
    def unload_data(self):
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)
        self.annual_constraints_data = np.empty((0,), dtype=np.float64)
        return None
    
    def availability_to_generation(self):
        if self.static_instance:
            raise_static_modification_error()
        self.data_status = "generation"
        if self.data.shape[0] > 0:
            self.data *= self.capacity
        return None

if JIT_ENABLED:
    storage_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order',int64),
        ('name',string),
        ('node',Node.class_type.instance_type),
        ('power_capacity',float64),
        ('energy_capacity',float64),
        ('duration',int64),
        ('charge_efficiency',float64),
        ('discharge_efficiency',float64),
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
    def __init__(self, 
                 static_instance,
                 idx,
                 order,
                 name,
                 power_capacity,
                 energy_capacity,
                 duration,
                 charge_efficiency,
                 discharge_efficiency,
                 max_build_p,
                 max_build_e,
                 min_build_p,
                 min_build_e,
                 unit_type,
                 near_optimum_check,
                 node,
                 line,
                 group,
                 cost,) -> None:
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

        self.static_instance = static_instance
        self.id = idx
        self.order = order # id specific to scenario
        self.name = name
        self.power_capacity = power_capacity  # GW
        self.energy_capacity = energy_capacity  # GWh
        self.duration = duration # hours
        self.charge_efficiency = charge_efficiency  # %
        self.discharge_efficiency = discharge_efficiency # %
        self.max_build_p = max_build_p  # GW/year
        self.max_build_e = max_build_e  # GWh/year
        self.min_build_p = min_build_p  # GW/year
        self.min_build_e = min_build_e  # GWh/year        
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check
        self.node = node
        self.line = line
        self.group = group            
        self.cost = cost

    def build_capacity(self, new_build_capacity, capacity_type):
        if self.static_instance:
            raise_static_modification_error()
        if capacity_type == "power":
            self.power_capacity += new_build_capacity
        if capacity_type == "energy":
            self.energy_capacity += new_build_capacity
        return None

if JIT_ENABLED:
    fleet_spec = [
        ('static_instance',boolean),
        ('generators', DictType(int64, Generator.class_type.instance_type)),
        ('storages', DictType(int64, Storage.class_type.instance_type)),
        ('traces', Traces2d.class_type.instance_type),
        ('generator_x_indices', int64[:]),
        ('storage_power_x_indices', int64[:]),
        ('storage_energy_x_indices', int64[:]),
    ]
else: 
    fleet_spec = []

@jitclass(fleet_spec)
class Fleet:
    def __init__(self,
                 static_instance,
                 generators,
                 storages,
                 traces,):
        self.static_instance = static_instance
        self.generators = generators
        self.storages = storages
        self.traces = traces

        self.generator_x_indices = np.full(len(generators), -1, dtype=np.int64)
        self.storage_power_x_indices = np.full(len(storages), -1, dtype=np.int64)
        self.storage_energy_x_indices = np.full(len(storages), -1, dtype=np.int64)

    def assign_x_indices(self, order, x_index, asset_type) -> None:
        if asset_type == 'generator':
            self.generator_x_indices[order] = x_index
        if asset_type == 'storage':
            self.storage_power_x_indices[order] = x_index
            self.storage_energy_x_indices[order] = x_index + len(self.storages)
        return None

    def build_capacities(self, decision_x) -> None:
        if self.static_instance:
            raise_static_modification_error()
            
        for order, index in enumerate(self.generator_x_indices):
            self.generators[order].build_capacity(decision_x[index])

        for order, index in enumerate(self.storage_power_x_indices):
            self.storages[order].build_capacity(decision_x[index], "power")

        for order, index in enumerate(self.storage_energy_x_indices):
            self.storages[order].build_capacity(decision_x[index], "energy")
        return None
    
    def get_generator_nodal_counts(self, node_count, unit_type):
        count_arr = np.zeros(node_count, dtype=np.int64)
        for generator in self.generators.values():
            if generator.unit_type == unit_type:
                count_arr[generator.node.order] += 1
        return count_arr
    
    def get_storage_nodal_counts(self, node_count):
        count_arr = np.zeros(node_count, dtype=np.int64)
        for storage in self.storages.values():
            count_arr[storage.node.order] += 1
        return count_arr