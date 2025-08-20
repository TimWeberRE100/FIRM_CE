import numpy as np
from typing import List, Tuple
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.costs import LTCosts, UnitCost_InstanceType, LTCosts_InstanceType
from firm_ce.common.typing import DictType, int64, UniTuple, ListType, float64, string, boolean
from firm_ce.common.jit_overload import jitclass

if JIT_ENABLED:
    node_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order',int64),
        ('name',string),
        ('data_status',string),
        ('data',float64[:]),

        ('residual_load',float64[:]), 

        # Dynamic
        ('storage_merit_order',int64[:]),
        ('flexible_merit_order',int64[:]),

        ('power_capacity',DictType(string,float64)), 
        ('energy_capacity',DictType(string,float64)), 

        ('netload_t',float64),
        ('discharge_max_t',float64[:]),
        ('charge_max_t',float64[:]),
        ('flexible_max_t',float64[:]),

        ('fill',float64),
        ('surplus',float64),
        ('available_imports',float64),

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

        # Dynamic
        self.flexible_merit_order = np.empty((0,), dtype=np.int64)
        self.storage_merit_order = np.empty((0,), dtype=np.int64)
        self.netload_t = 0.0 # GW
        self.discharge_max_t = np.empty((0,), dtype=np.float64) # GW
        self.charge_max_t = np.empty((0,), dtype=np.float64) # GW
        self.flexible_max_t = np.empty((0,), dtype=np.float64) # GW

        self.fill = 0.0 # GW, power attempting to import
        self.surplus = 0.0 # GW, power available for exports
        self.available_imports = 0.0 # GW, maximum power that could be imported from other node surpluses

        self.imports = np.empty((0,), dtype=np.float64)
        self.exports = np.empty((0,), dtype=np.float64)
        self.deficits = np.empty((0,), dtype=np.float64)
        self.spillage = np.empty((0,), dtype=np.float64)

        self.flexible_power = np.empty((0,), dtype=np.float64)
        self.storage_power = np.empty((0,), dtype=np.float64)
        self.flexible_energy = np.empty((0,), dtype=np.float64)
        self.storage_energy = np.empty((0,), dtype=np.float64) 

if JIT_ENABLED:
    Node_InstanceType = Node.class_type.instance_type
else:
    Node_InstanceType = Node

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
        ('initial_capacity', float64),
        ('unit_type', string),
        ('near_optimum_check', boolean),
        ('group', string),
        ('cost', UnitCost_InstanceType),

        ('candidate_x_idx',int64),

        # Dynamic
        ('new_build',float64),
        ('capacity', float64),
        ('flows',float64[:]),
        ('temp_leg_flows', float64),
        ('lt_flows', float64),

        ('lt_costs',LTCosts_InstanceType),
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
        self.initial_capacity = capacity  # GW  
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check
        self.group = group
        self.cost = cost

        self.candidate_x_idx = -1

        # Dynamic
        self.new_build = 0.0 # GW
        self.capacity = capacity # GW
        self.flows = np.empty(0, dtype=np.float64) # GW, total line flows
        self.temp_leg_flows = 0.0 # GW, line flows reserved for a route on the current leg
        self.lt_flows = 0.0 # GWh

        self.lt_costs = LTCosts()

if JIT_ENABLED:
    Line_InstanceType = Line.class_type.instance_type
else:
    Line_InstanceType = Line

if JIT_ENABLED:
    route_spec = [
        ('static_instance',boolean),
        ('initial_node',Node_InstanceType),
        ('nodes',ListType(Node_InstanceType)),
        ('lines',ListType(Line_InstanceType)),   
        ('line_directions',int64[:]),
        ('legs',int64),
        ('flow_update',float64),
    ]
else:
    route_spec = []

@jitclass(route_spec)
class Route:
    def __init__(self,
                 static_instance,
                 initial_node,
                 nodes_typed_list,
                 lines_typed_list,
                 line_directions,
                 legs,):
        self.static_instance = static_instance
        self.initial_node = initial_node
        self.nodes = nodes_typed_list
        self.lines = lines_typed_list
        self.line_directions = line_directions
        self.legs = legs

        # Dynamic
        self.flow_update = 0.0

if JIT_ENABLED:
    Route_InstanceType = Route.class_type.instance_type
    routes_key_type = UniTuple(int64,2)
    routes_list_type = ListType(Route_InstanceType)
else:
    Route_InstanceType = Route
    routes_key_type = Tuple[int,int]
    routes_list_type = List[Route_InstanceType]

if JIT_ENABLED:
    network_spec = [
        ('static_instance',boolean),
        ('nodes',DictType(int64,Node_InstanceType)),
        ('major_lines',DictType(int64,Line_InstanceType)),
        ('minor_lines',DictType(int64,Line_InstanceType)),
        ('networksteps_max', int64),
        ('routes',DictType(routes_key_type, routes_list_type)), # Key is Tuple(initial_node.order, legs)
        ('major_line_count',int64),
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
                 routes,
                 networksteps_max,
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
        self.routes = routes
        self.networksteps_max = networksteps_max
        self.major_line_count = len(major_lines)

if JIT_ENABLED:  
    Network_InstanceType = Network.class_type.instance_type
else:
    Network_InstanceType = Network