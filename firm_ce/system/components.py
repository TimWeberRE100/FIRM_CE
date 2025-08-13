import numpy as np
from numpy.typing import NDArray
from typing import Union

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.exceptions import (
    raise_static_modification_error,
    raise_getting_unloaded_data_error,
)
from firm_ce.system.costs import LTCosts, UnitCost_InstanceType, LTCosts_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType

if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType
    from numba.experimental import jitclass
    from numba.typed.typeddict import Dict as TypedDict

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

Fuel_InstanceType = Fuel.class_type.instance_type

if JIT_ENABLED:
    generator_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order', int64),
        ('name',string),
        ('node',Node_InstanceType),
        ('fuel',Fuel_InstanceType),
        ('unit_size',float64),
        ('max_build',float64),
        ('min_build',float64),
        ('initial_capacity',float64),
        ('line',Line_InstanceType),
        ('unit_type',string),
        ('near_optimum_check',boolean),
        ('group',string),
        ('cost',UnitCost_InstanceType),
        ('data_status',string),
        ('data',float64[:]),
        ('annual_constraints_data',float64[:]),

        ('candidate_x_idx',int64),

        # Dynamic
        ('new_build',float64),
        ('capacity',float64),
        ('dispatch_power',float64[:]),
        ('remaining_energy',float64[:]),
        ('flexible_max_t',float64),   

        ('lt_generation',float64),
        ('unit_lt_hours',float64),

        ('lt_costs', LTCosts_InstanceType),
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
        self.initial_capacity = capacity  # GW        
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

        self.candidate_x_idx = -1

        # Dynamic
        self.new_build = 0.0 # GW
        self.capacity = capacity  # GW 
        self.dispatch_power = np.empty((0,), dtype=np.float64) # GW
        self.remaining_energy = np.empty((0,), dtype=np.float64) # GWh

        self.flexible_max_t = 0.0 # GW
        self.lt_generation = 0.0 # GWh
        self.unit_lt_hours = 0.0 # hours/unit

        self.lt_costs = LTCosts()
        
    def create_dynamic_copy(self, nodes_typed_dict, lines_typed_dict):
        node_copy = nodes_typed_dict[self.node.order]
        line_copy = lines_typed_dict[self.line.order] 

        generator_copy = Generator(
            False,
            self.id, 
            self.order,
            self.name,
            self.unit_size,
            self.max_build,
            self.min_build,
            self.capacity,
            self.unit_type,
            self.near_optimum_check,
            node_copy,
            self.fuel, # This remains static
            line_copy,
            self.group,
            self.cost, # This remains static
        )
        generator_copy.data_status = self.data_status
        generator_copy.data = self.data # This remains static
        generator_copy.annual_constraints_data = self.annual_constraints_data # This remains static
        generator_copy.candidate_x_idx = self.candidate_x_idx
        generator_copy.lt_generation = self.lt_generation

        return generator_copy

    def build_capacity(self, new_build_power_capacity: float, resolution: float):
        if self.static_instance:
            raise_static_modification_error()     
        self.capacity += new_build_power_capacity   
        self.new_build += new_build_power_capacity 
        self.node.power_capacity[self.unit_type] += new_build_power_capacity
        self.line.capacity += new_build_power_capacity 

        self.update_residual_load(new_build_power_capacity, resolution)     
        return None
    
    def load_data(self, generation_trace: NDArray[np.float64], annual_constraints: NDArray[np.float64], resolution: float):
        self.data_status= "loaded"
        self.data = generation_trace
        self.annual_constraints_data = annual_constraints

        self.update_residual_load(self.initial_capacity, resolution)
        return None
    
    def unload_data(self):
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)
        self.annual_constraints_data = np.empty((0,), dtype=np.float64)
        return None
    
    def get_data(self, data_type: str) -> Union[NDArray[np.float64], None]:
        if self.data_status == "unloaded":
            raise_getting_unloaded_data_error()
        
        if data_type == "annual_constraints_data":
            return self.annual_constraints_data        
        elif data_type == "trace":
            return self.data        
        else:
            raise RuntimeError("Invalid data_type argument for Generator.get_data(data_type).")
        return None

    def allocate_memory(self, intervals_count):
        if self.static_instance:
            raise_static_modification_error()
        self.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
        if len(self.get_data('annual_constraints_data')) > 0:
            self.remaining_energy = np.zeros(intervals_count, dtype=np.float64)
        return None
    
    def update_residual_load(self, added_capacity: float, resolution: float) -> None:
        if self.get_data("trace").shape[0] > 0 and added_capacity > 0.0:
            new_trace = self.get_data("trace") * added_capacity
            self.node.get_data("residual_load")[:] -= new_trace
            self.update_lt_generation(new_trace, resolution) 
        return None
    
    def update_lt_generation(self, generation_trace: NDArray[np.float64], resolution: float) -> None:
        self.lt_generation += sum(generation_trace) * resolution
        self.line.lt_flows += self.lt_generation
        return None
    
    def initialise_annual_limit(self, year, first_t) :
        if len(self.get_data('annual_constraints_data')) > 0:
            self.remaining_energy[first_t-1] = self.get_data('annual_constraints_data')[year]
        return None
    
    def check_unit_type(self, unit_type: str) -> bool:
        return self.unit_type == unit_type
    
    def set_flexible_max_t(self, interval: int, resolution: float, merit_order_idx: int) -> None:
        self.flexible_max_t = min(
            self.capacity, 
            self.remaining_energy[interval-1] / resolution
        )
        self.node.flexible_max_t[merit_order_idx] = self.node.flexible_max_t[merit_order_idx-1] + self.flexible_max_t
        return None
    
    def dispatch(self, interval: int, merit_order_idx: int) -> bool:
        if merit_order_idx == 0:
            self.dispatch_power[interval] = min(
                max(self.node.netload_t - self.node.storage_power[interval], 0.0),
                self.flexible_max_t
            )
        else:
            self.dispatch_power[interval] = min(
                max(self.node.netload_t - self.node.storage_power[interval] - self.node.flexible_max_t[merit_order_idx-1], 0.0),
                self.flexible_max_t
            )
        self.node.flexible_power[interval] += self.dispatch_power[interval]
        return self.node.check_remaining_netload(interval, 'deficit')
    
    def update_remaining_energy(self, interval: int, resolution: float) -> None:
        self.remaining_energy[interval] = self.remaining_energy[interval-1] - self.dispatch_power[interval] * resolution
        return None
    
    def calculate_lt_generation(self, resolution: float) -> None:
        self.update_lt_generation(self.dispatch_power, resolution)
        self.unit_lt_hours = sum(np.ceil(self.dispatch_power/self.unit_size)) * resolution
        return None
    
    def calculate_lt_costs(self, years_float: float) -> float:
        self.lt_costs.calculate_annualised_build(0.0, self.capacity, 0.0, self.cost, 'generator')
        self.lt_costs.calculate_fom(self.capacity, years_float, 0.0, self.cost, 'generator')
        self.lt_costs.calculate_vom(self.lt_generation, self.cost)
        self.lt_costs.calculate_fuel(self.lt_generation, self.unit_lt_hours, self.cost)
        return self.lt_costs.get_total()

Generator_InstanceType = Generator.class_type.instance_type 

if JIT_ENABLED:
    storage_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order',int64),
        ('name',string),
        ('node',Node_InstanceType),
        ('initial_power_capacity',float64),
        ('initial_energy_capacity',float64),
        ('duration',int64),
        ('charge_efficiency',float64),
        ('discharge_efficiency',float64),
        ('max_build_p',float64),
        ('max_build_e',float64),
        ('min_build_p',float64),
        ('min_build_e',float64),        
        ('line',Line_InstanceType),
        ('unit_type',string),
        ('near_optimum_check',boolean),
        ('group',string),
        ('cost',UnitCost_InstanceType),

        ('candidate_p_x_idx',int64),
        ('candidate_e_x_idx',int64),

        # Dynamic
        ('new_build_p',float64),
        ('new_build_e',float64),
        ('power_capacity',float64),
        ('energy_capacity',float64),
        ('dispatch_power',float64[:]),
        ('stored_energy',float64[:]),

        ('discharge_max_t',float64),
        ('charge_max_t',float64),
        ('lt_discharge',float64),

        ('lt_costs',LTCosts_InstanceType),
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
        self.initial_power_capacity = power_capacity  # GW
        self.duration = duration # hours
        self.initial_energy_capacity = energy_capacity if duration == 0 else duration*power_capacity # GWh
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

        self.candidate_p_x_idx = -1
        self.candidate_e_x_idx = -1

        # Dynamic
        self.new_build_p = 0.0 # GW
        self.new_build_e = 0.0 # GWh
        self.power_capacity = power_capacity # GW
        self.energy_capacity = energy_capacity if duration == 0 else duration*power_capacity  # GWh
        self.dispatch_power = np.empty(0, dtype=np.float64) # GW
        self.stored_energy = np.empty(0, dtype=np.float64) # GWh

        self.discharge_max_t = 0.0 # GW
        self.charge_max_t = 0.0 # GW
        self.lt_discharge = 0.0 # GWh/year

        self.lt_costs = LTCosts()

    def create_dynamic_copy(self, nodes_typed_dict, lines_typed_dict):
        node_copy = nodes_typed_dict[self.node.order]
        line_copy = lines_typed_dict[self.line.order] 

        storage_copy = Storage(
            False,
            self.id,
            self.order,
            self.name,
            self.power_capacity,
            self.energy_capacity,
            self.duration,
            self.charge_efficiency,
            self.discharge_efficiency,
            self.max_build_p,
            self.max_build_e,
            self.min_build_p,
            self.min_build_e,
            self.unit_type,
            self.near_optimum_check,
            node_copy,
            line_copy,
            self.group, 
            self.cost, # This remains static
        )

        storage_copy.candidate_p_x_idx = self.candidate_p_x_idx
        storage_copy.candidate_e_x_idx = self.candidate_e_x_idx    
        
        return storage_copy

    def build_capacity(self, new_build_capacity, capacity_type):
        if self.static_instance:
            raise_static_modification_error()
        if capacity_type == "power":
            self.power_capacity += new_build_capacity
            self.new_build_p += new_build_capacity
            self.node.power_capacity['storage'] += new_build_capacity
            self.line.capacity += new_build_capacity

            if self.duration > 0:
                self.energy_capacity += new_build_capacity * self.duration
                self.new_build_e += new_build_capacity * self.duration
                self.node.energy_capacity['storage'] += new_build_capacity * self.duration
                
        if capacity_type == "energy":
            if self.duration == 0:
                self.energy_capacity += new_build_capacity 
                self.new_build_e += new_build_capacity
                self.node.energy_capacity['storage'] += new_build_capacity
        return None
    
    def allocate_memory(self, intervals_count):
        if self.static_instance:
            raise_static_modification_error()
        self.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
        self.stored_energy = np.zeros(intervals_count, dtype=np.float64)
        return None

    def initialise_stored_energy(self):
        if self.static_instance:
            raise_static_modification_error()
        self.stored_energy[-1] = 0.5*self.energy_capacity
        return None
    
    def set_dispatch_max_t(self, interval: int, resolution: float, merit_order_idx: int):
        self.discharge_max_t = min(
            self.power_capacity, 
            self.stored_energy[interval-1] * self.discharge_efficiency / resolution
        )
        self.charge_max_t = min(
            self.power_capacity, 
            (self.energy_capacity - self.stored_energy[interval-1]) / self.charge_efficiency / resolution
        )

        self.node.discharge_max_t[merit_order_idx] = self.node.discharge_max_t[merit_order_idx-1] + self.discharge_max_t
        self.node.charge_max_t[merit_order_idx] = self.node.charge_max_t[merit_order_idx-1] + self.charge_max_t
        return None
    
    def dispatch(self, interval: int, merit_order_idx: int) -> bool:
        if merit_order_idx == 0:
            self.dispatch_power[interval] = (
                max(min(self.node.netload_t, self.discharge_max_t), 0.0) +
                min(max(self.node.netload_t, -self.charge_max_t), 0.0)
            )
        else:
            self.dispatch_power[interval] = (
                min(max(0, self.node.netload_t - self.node.discharge_max_t[merit_order_idx-1]), self.discharge_max_t) +
                max(min(0, self.node.netload_t + self.node.charge_max_t[merit_order_idx-1]), -self.charge_max_t)
            )
        self.node.storage_power[interval] += self.dispatch_power[interval]
        return self.node.check_remaining_netload(interval, 'both')
    
    def update_stored_energy(self, interval: int, resolution: float) -> None:
        self.stored_energy[interval] = self.stored_energy[interval-1] \
            - max(self.dispatch_power[interval], 0) / self.discharge_efficiency * resolution \
            - min(self.dispatch_power[interval], 0) * self.charge_efficiency * resolution 
        return None
    
    def calculate_lt_discharge(self, resolution: float) -> None:
        self.lt_discharge = sum(
            np.maximum(self.dispatch_power, 0)
        ) * resolution

        self.line.lt_flows += sum(
            np.abs(self.dispatch_power)
        ) * resolution
        return None
    
    def calculate_lt_costs(self, years_float: float) -> float:
        self.lt_costs.calculate_annualised_build(self.energy_capacity, self.power_capacity, 0.0, self.cost, 'storage')
        self.lt_costs.calculate_fom(self.power_capacity, years_float, 0.0, self.cost, 'storage')
        self.lt_costs.calculate_vom(self.lt_discharge, self.cost)
        self.lt_costs.calculate_fuel(self.lt_discharge, 0, self.cost)
        return self.lt_costs.get_total()
    
Storage_InstanceType = Storage.class_type.instance_type

if JIT_ENABLED:
    fleet_spec = [
        ('static_instance',boolean),
        ('generators', DictType(int64, Generator_InstanceType)),
        ('storages', DictType(int64, Storage_InstanceType)),
    ]
else: 
    fleet_spec = []

@jitclass(fleet_spec)
class Fleet:
    def __init__(self,
                 static_instance,
                 generators,
                 storages,):
        self.static_instance = static_instance
        self.generators = generators
        self.storages = storages

    def create_dynamic_copy(self, nodes_typed_dict, lines_typed_dict):
        generators_copy = TypedDict.empty(
            key_type=int64,
            value_type=Generator_InstanceType
        )
        storages_copy = TypedDict.empty(
            key_type=int64,
            value_type=Storage_InstanceType
        )

        for order, generator in self.generators.items():
            generators_copy[order] = generator.create_dynamic_copy(nodes_typed_dict, lines_typed_dict)

        for order, storage in self.storages.items():
            storages_copy[order] = storage.create_dynamic_copy(nodes_typed_dict, lines_typed_dict)

        fleet_copy = Fleet(False,
                     generators_copy,
                     storages_copy,)
        
        return fleet_copy

    def build_capacities(self, decision_x, resolution: float) -> None:
        if self.static_instance:
            raise_static_modification_error()
            
        for generator in self.generators.values():
            generator.build_capacity(decision_x[generator.candidate_x_idx], resolution)

        for storage in self.storages.values():
            storage.build_capacity(decision_x[storage.candidate_p_x_idx], "power")
            storage.build_capacity(decision_x[storage.candidate_e_x_idx], "energy")
        return None
    
    def allocate_memory(self, intervals_count):
        if self.static_instance:
            raise_static_modification_error()

        for generator in self.generators.values():
            if generator.unit_type == 'flexible':
                generator.allocate_memory(intervals_count)

        for storage in self.storages.values():
            storage.allocate_memory(intervals_count)
        
        return None
    
    def initialise_stored_energies(self):
        if self.static_instance:
            raise_static_modification_error()
        for storage in self.storages.values():
            storage.initialise_stored_energy()
        return None
    
    def initialise_annual_limits(self, year: int, first_t: int):
        if self.static_instance:
            raise_static_modification_error()        
        for generator in self.generators.values():            
            generator.initialise_annual_limit(year, first_t)        
        return None
    
    def count_generator_unit_type(self, unit_type: str) -> int:
        count = 0
        for generator in self.generators.values():
            if generator.unit_type == unit_type:
                count+=1
        return count
    
    def update_stored_energies(self, interval: int, resolution: float) -> None:
        for storage in self.storages.values():
            storage.update_stored_energy(interval, resolution)
        return None
    
    def update_remaining_flexible_energies(self, interval: int, resolution: float) -> None:
        for generator in self.generators.values():
            if not generator.check_unit_type('flexible'):
                continue
            generator.update_remaining_energy(interval, resolution)
        return None
    
    def calculate_lt_generations(self, resolution: float) -> None:
        for generator in self.generators.values():
            if generator.check_unit_type('flexible'):
                generator.calculate_lt_generation(resolution)

        for storage in self.storages.values():
            storage.calculate_lt_discharge(resolution)
        return None

Fleet_InstanceType = Fleet.class_type.instance_type