import numpy as np
from numpy.typing import NDArray

from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.costs import UnitCost
from firm_ce.common.helpers import array_min, array_max_2d_axis1, array_sum_2d_axis0, zero_safe_division

if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType, UniTuple
    from numba.experimental import jitclass

    node_spec = [
        ('static_instance',boolean),
        ('id',int64),
        ('order',int64),
        ('name',string),
        ('data_status',string),
        ('data',float64[:]),
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

    def load_data(self, trace):
        self.data_status= "demand"
        self.data = trace
        return None
    
    def unload_data(self):
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)
        return None

if JIT_ENABLED:
    line_spec = [
        ('static_instance',boolean),
        ('id', int64),
        ('order', int64),
        ('name', string),
        ('length', float64),
        ('node_start', Node.class_type.instance_type),
        ('node_end', Node.class_type.instance_type),
        ('loss_factor', float64),
        ('max_build', float64),
        ('min_build', float64),
        ('capacity', float64),
        ('unit_type', string),
        ('near_optimum_check', boolean),
        ('group', string),
        ('cost', UnitCost.class_type.instance_type),
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

    def check_minor_line(self) -> bool:
        return self.id == -1
    
    def build_capacity(self, new_build_power_capacity):
        if self.static_instance:
            raise_static_modification_error()
        self.capacity += new_build_power_capacity
        return None

if JIT_ENABLED:
    network_spec = [
        ('static_instance',boolean),
        ('nodes',DictType(int64,Node.class_type.instance_type)),
        ('major_lines',DictType(int64,Line.class_type.instance_type)),
        ('minor_lines',DictType(int64,Line.class_type.instance_type)),        
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
    
    #### THIS NEEDS TO BE MOVED OUT OF STATIC INSTANCE
    def get_transmission_flows_t(self,
                                Fillt: NDArray[np.float64], 
                                Surplust: NDArray[np.float64], 
                                Importt: NDArray[np.float64], 
                                Exportt: NDArray[np.float64],
                                ) -> NDArray[np.float64]:
        '''Improve efficiency by avioding so many new variable declarations'''
        # The primary connections are simpler (and faster) to model than the general
        #   nthary connection
        # Since many if not most calls of this function only require primary transmission
        #   I have split it out from general nthary transmission to improve speed
        _transmission = np.zeros(len(self.nodes), np.float64)
        leg = 0
        # loop through nodes with deficits
        for n in self.nodes:
            if Fillt[n] < 1e-6:
                continue
            # appropriate slice of network array
            # pdonors is equivalent to donors later on but has different ndim so needs to
            #   be a different variable name for static typing
            pdonors, pdonor_lines = self.cache_0_donors[n]
            _usage = 0.0 # badly named by avoids creating more variables
            for d in pdonors: 
                _usage += Surplust[d]

            if _usage < 1e-6:
                # continue if no surplus to be traded
                continue

            for d, l in zip(pdonors, pdonor_lines):
                _usage = 0.0
                for m in self.nodes:
                    _usage += Importt[m, l]
                # maximum exportable
                _transmission[d] = min(
                    Surplust[d],  # power resource constraint
                    self.transmission_capacities[l] - _usage, # line capacity constraint
                )  

            # scale down to fill requirement
            _usage = 0.0
            for m in self.nodes:
                _usage += _transmission[m] 
            if _usage > Fillt[n]:
                _scale = Fillt[n] / _usage
                _transmission *= _scale
                _usage *= _scale
            if _usage < 1e-6:
                continue

            # for d, l in zip(pdonors, pdonor_lines):  #  print(d,l)
            for i in range(len(pdonors)):
                # record transmission
                Importt[n, pdonor_lines[i]] += _transmission[pdonors[i]]
                Exportt[pdonors[i], pdonor_lines[i]] -= _transmission[pdonors[i]]
                # adjust deficit/surpluses
                Surplust[pdonors[i]] -= _transmission[pdonors[i]]
                _transmission[pdonors[i]] = 0
    
            Fillt[n] -= _usage

        # Continue with nthary transmission
        # Note: This code block works for primary transmission too, but is slower
        if (Fillt.sum() > 1e-6) and (Surplust.sum() > 1e-6):
                _import = np.zeros(Importt.shape, np.float64)
                _capacity = np.zeros(self.major_line_count, np.float64)
                # loop through secondary, tertiary, ..., nthary connections
                for leg in range(1, self.networksteps_max):
                    # loop through nodes with deficits
                    for n in self.nodes:
                        if Fillt[n] < 1e-6:
                            continue
                        
                        donors, donor_lines = self.cache_n_donors[(n, leg)]

                        if donors.shape[1] == 0:
                            break  # break if no valid donors
                            
                        _usage = 0.0 # badly named variable but avoids extra variables
                        for d in donors[-1]:
                            _usage += Surplust[d]

                        if _usage < 1e-6:
                            continue

                        _capacity[:] = self.transmission_capacities - array_sum_2d_axis0(Importt)
                        for d, dl in zip(donors[-1], donor_lines.T): # print(d,dl)
                            # power use of each line, clipped to maximum capacity of lowest leg
                            _import[d, dl] = min(array_min(_capacity[dl]), Surplust[d])
                        
                        for l in range(self.major_line_count):
                            # total usage of the line across all import paths
                            _usage=0.0
                            for m in self.nodes:
                                _usage += _import[m, l]
                            # if usage exceeds capacity
                            if _usage > _capacity[l]:
                                # unclear why this raises zero division error from time to time
                                _scale = zero_safe_division(_capacity[l], _usage)
                                for m in self.nodes:
                                    # clip all legs
                                    if _import[m, l] > 1e-6:
                                        for o in self.major_lines:
                                            _import[m, o] *= _scale
                            
                        # intermediate calculation array
                        _transmission = array_max_2d_axis1(_import)
                        
                        # scale down to fill requirement
                        _usage = 0.0
                        for m in self.nodes:
                            _usage += _transmission[m] 
                        if _usage > Fillt[n]:
                            _scale = Fillt[n] / _usage
                            _transmission *= _scale
                            _usage *= _scale
                        if _usage < 1e-6:
                            continue

                        for nd, d, dl in zip(range(donors.shape[1]), donors[-1], donor_lines.T): # print(nd, d, dl)
                            Importt[n, dl[0]] += _transmission[d]
                            Exportt[donors[0, nd], dl[0]] -= _transmission[d]
                            for step in range(leg):
                                Importt[donors[step, nd], dl[step+1]] += _transmission[d]
                                Exportt[donors[step+1, nd], dl[step+1]] -= _transmission[d]

                        # Adjust fill and surplus
                        Fillt[n] -= _usage
                        Surplust -= _transmission
                        
                        _import[:] = 0.0
                        _capacity[:] = 0.0
                        
                        if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                            break

                    if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                        break

        return Importt, Exportt