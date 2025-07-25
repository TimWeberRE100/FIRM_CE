import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import (
    raise_static_modification_error,
    raise_getting_unloaded_data_error,
)
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.costs import UnitCost_InstanceType
#from firm_ce.system.components import Generator_InstanceType

if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType, UniTuple
    from numba.experimental import jitclass
    from numba.typed.typeddict import Dict as TypedDict

    node_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order',int64),
        ('name',string),
        ('data_status',string),
        ('data',float64[:]),

        ('residual_load',float64[:]), 
        ('flexible_p_capacity',float64), 
        ('storage_p_capacity',float64), 
        ('storage_e_capacity',float64),

        # Dynamic
        ('imports',float64[:]),
        ('exports',float64[:]), 
        ('deficits',float64[:]), 
        ('spillage',float64[:]), 

        ('flexible_power',float64[:]), 
        ('storage_power',float64[:]),
        ('flexible_energy',float64[:]), 
        ('storage_energy',float64[:]), 
    ]
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    node_spec = []

@jitclass(node_spec)
class Node:
    """
    Represents a node (bus) in the network. All nodes require a demand trace
    stored in a datafile within 'data' and referenced in 'config/datafiles.csv'. 
    """

    def __init__(self, static_instance, idx, order, name) -> None:
        """
        Initialize a Node.

        Parameters:
        -------
        id (int): Unique identifier for the node.
        name (str): Name of the node.
        """
        self.static_instance = static_instance
        self.id = idx
        self.order = order # id specific to scenario
        self.name = name
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)

        self.residual_load = np.empty((0,), dtype=np.float64)
        self.flexible_p_capacity = 0.0
        self.storage_p_capacity = 0.0
        self.storage_e_capacity = 0.0

        # Dynamic
        self.imports = np.empty((0,), dtype=np.float64)
        self.exports = np.empty((0,), dtype=np.float64)
        self.deficits = np.empty((0,), dtype=np.float64)
        self.spillage = np.empty((0,), dtype=np.float64)

        self.flexible_power = np.empty((0,), dtype=np.float64)
        self.storage_power = np.empty((0,), dtype=np.float64)
        self.flexible_energy = np.empty((0,), dtype=np.float64)
        self.storage_energy = np.empty((0,), dtype=np.float64)        

    def create_dynamic_copy(self):
        node_copy = Node(
            False,
            self.id,
            self.order,
            self.name
        )
        node_copy.data_status = self.data_status # This remains static
        node_copy.data = self.data # This remains static
        node_copy.residual_load = self.residual_load # This remains static
        return node_copy        

    def load_data(self, 
                  trace: NDArray[np.float64],):
        self.data_status = "demand"
        self.data = trace
        return None
    
    def unload_data(self):
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)
        return None
    
    def get_data(self):
        if self.data_status == "unloaded":
            raise_getting_unloaded_data_error()
        return self.data
    
    def initialise_residual_load(
            self,
            generators_typed_dict, # TypedDict[int64, Generator_InstanceType]
            intervals_count: int,) -> None:
        self.residual_load = self.get_data() / 1000 # Convert to GW

        for generator in generators_typed_dict.values():
            if (generator.data_status == "availability") and (generator.node.order == self.order):                
                for interval in range(intervals_count):
                    self.residual_load[interval] -= generator.get_data("trace")[interval] * generator.initial_capacity  
        return None
    
    def update_residual_load(self,
                             generators_typed_dict, # TypedDict[int64, Generator_InstanceType]
                             intervals_count: int,):
        if self.static_instance:
            raise_static_modification_error()
        return None
    
    def initialise_nodal_capacities(self):
        if self.static_instance:
            raise_static_modification_error()
        return None
    
    def allocate_memory(self):
        if self.static_instance:
            raise_static_modification_error()
        self.imports = np.zeros_like(self.residual_load, dtype=np.float64)
        self.exports = np.zeros_like(self.residual_load, dtype=np.float64)
        self.deficits = np.zeros_like(self.residual_load, dtype=np.float64)
        self.spillage = np.zeros_like(self.residual_load, dtype=np.float64)

        self.flexible_power = np.zeros_like(self.residual_load, dtype=np.float64)
        self.storage_power = np.zeros_like(self.residual_load, dtype=np.float64)
        self.flexible_energy = np.zeros_like(self.residual_load, dtype=np.float64)
        self.storage_energy = np.zeros_like(self.residual_load, dtype=np.float64)
        return None

Node_InstanceType = Node.class_type.instance_type

if JIT_ENABLED:
    line_spec = [
        ('static_instance',boolean),
        ('id', int64),
        ('order', int64),
        ('name', string),
        ('length', float64),
        ('node_start', Node_InstanceType),
        ('node_end', Node_InstanceType),
        ('loss_factor', float64),
        ('max_build', float64),
        ('min_build', float64),
        ('capacity', float64),
        ('unit_type', string),
        ('near_optimum_check', boolean),
        ('group', string),
        ('cost', UnitCost_InstanceType),
    ]
else:
    line_spec = []

@jitclass(line_spec)
class Line:
    """
    Represents a transmission line connecting two nodes within the network.

    If node_start and node_end values are empty (np.nan) then the line is a
    minor line that instead connects a generator or storage object to the
    transmission network.
    """
    def __init__(self, 
                 static_instance,
                 idx,
                 order,
                 name,
                 length,
                 node_start,
                 node_end,
                 loss_factor,
                 max_build,
                 min_build,
                 capacity,
                 unit_type,
                 near_optimum_check,
                 group,
                 cost,) -> None:
        """
        Initialize a Line object.
        """
        self.static_instance = static_instance
        self.id = idx
        self.order = order # id specific to scenario
        self.name = name
        self.length = length # km
        self.node_start = node_start # Starting node
        self.node_end = node_end # Ending node
        self.loss_factor = loss_factor # Transmission losses % per 1000 km
        self.max_build = max_build # GW/year
        self.min_build = min_build # GW/year
        self.capacity = capacity # GW
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check
        self.group = group
        self.cost = cost

    def create_dynamic_copy(self, nodes_typed_dict, line_type):
        if line_type == "major":
            node_start_copy = nodes_typed_dict[self.node_start.order]
            node_end_copy = nodes_typed_dict[self.node_end.order]
        elif line_type == "major":
            node_start_copy = Node(False,-1,-1,"MINOR_NODE")
            node_end_copy = Node(False,-1,-1,"MINOR_NODE")
        
        line_copy = Line(
            False,
            self.id,
            self.order,
            self.name,
            self.length,
            node_start_copy,
            node_end_copy,
            self.loss_factor,
            self.max_build,
            self.min_build,
            self.capacity,
            self.unit_type,
            self.near_optimum_check,
            self.group,
            self.cost, # This remains static
        )
        return line_copy

    def check_minor_line(self) -> bool:
        return self.id == -1
    
    def build_capacity(self, new_build_power_capacity):
        if self.static_instance:
            raise_static_modification_error()
        self.capacity += new_build_power_capacity
        return None

Line_InstanceType = Line.class_type.instance_type

if JIT_ENABLED:
    network_spec = [
        ('static_instance',boolean),
        ('nodes',DictType(int64,Node_InstanceType)),
        ('major_lines',DictType(int64,Line_InstanceType)),
        ('minor_lines',DictType(int64,Line_InstanceType)),        
        ('cache_0_donors',DictType(int64, int64[:, :])),
        ('cache_n_donors',DictType(UniTuple(int64, 2), int64[:, :, :])),
        ('transmission_mask',boolean[:,:]),
        ('networksteps_max', int64),
        ('transmission_capacities', float64[:]),
        ('major_line_count',int64),
        ('line_x_indices',int64[:]),
    ]
else:
    network_spec = []

@jitclass(network_spec)
class Network:
    """
    Constructs the network topology for transmission modeling using lines and nodes.
    Provides access to transmission masks, direct connection matrices, and nth-order networks
    required for transmission business rules in the unit commitment problem.
    """

    def __init__(self, 
                 static_instance,
                 nodes,
                 major_lines,
                 minor_lines,
                 cache_0_donors,
                 cache_n_donors,
                 transmission_mask,
                 networksteps_max,
                 transmission_capacities_initial,
                 ) -> None:
        """
        Initialize the Network topology and build all relevant matrices and masks.

        Parameters:
        -------
        lines (Dict[int, Line]): Dictionary of transmission lines.
        nodes (Dict[int, Node]): Dictionary of nodes in the system.
        networksteps_max (int): Maximum number of legs along which transmission can occur.
        """
        self.static_instance = static_instance
        self.nodes = nodes
        self.major_lines = major_lines
        self.minor_lines = minor_lines
        self.cache_0_donors = cache_0_donors
        self.cache_n_donors = cache_n_donors
        self.transmission_mask = transmission_mask
        self.networksteps_max = networksteps_max
        self.transmission_capacities = transmission_capacities_initial
        self.major_line_count = len(major_lines)
        self.line_x_indices = np.full(len(major_lines), -1, dtype=np.int64)

    def create_dynamic_copy(self):
        nodes_copy = TypedDict.empty(
            key_type=int64,
            value_type=Node_InstanceType
        )
        major_lines_copy = TypedDict.empty(
            key_type=int64,
            value_type=Line_InstanceType
        )
        minor_lines_copy = TypedDict.empty(
            key_type=int64,
            value_type=Line_InstanceType
        )

        for order, node in self.nodes.items():
            nodes_copy[order] = node.create_dynamic_copy()

        for order, line in self.major_lines.items():
            major_lines_copy[order] = line.create_dynamic_copy(nodes_copy, "major")

        for order, line in self.minor_lines.items():
            minor_lines_copy[order] = line.create_dynamic_copy(nodes_copy, "minor")

        network_copy = Network(
            False,
            nodes_copy,
            major_lines_copy,
            minor_lines_copy,
            self.cache_0_donors, # This is static
            self.cache_n_donors, # This is static
            self.transmission_mask, # This is static
            self.networksteps_max,
            self.transmission_capacities.copy(),
        )
        network_copy.major_line_count = self.major_line_count
        network_copy.line_x_indices = self.line_x_indices # This is static
        return network_copy

    def assign_x_indices(self, order, x_index) -> None:
        self.line_x_indices[order] = x_index
        return None

    def build_capacity(self, decision_x) -> None:
        if self.static_instance:
            raise_static_modification_error()
        for order, index in enumerate(self.line_x_indices):
            self.lines[order].capacity += decision_x[index]
        return None
    
    def unload_data(self):
        for node in self.nodes.values():
            node.unload_data()
        return None
    
    def allocate_memory(self):
        if self.static_instance:
            raise_static_modification_error()
        for node in self.nodes.values():
            node.allocate_memory()
        return None
    
    def update_residual_loads(self,
                             generators_typed_dict, # TypedDict[int64, Generator_InstanceType]
                             intervals_count: int,):
        if self.static_instance:
            raise_static_modification_error()
        for node in self.nodes.values():
            node.update_residual_load(
                generators_typed_dict,
                intervals_count
            )
        return None
    
Network_InstanceType = Network.class_type.instance_type