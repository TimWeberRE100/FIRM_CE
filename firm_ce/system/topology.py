from typing import Dict
import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.io.file_manager import DataFile 
from firm_ce.system.costs import UnitCost
from firm_ce.io.validate import is_nan
from firm_ce.common.helpers import array_min, array_max_2d_axis1, array_sum_2d_axis0, zero_safe_division

if JIT_ENABLED:
    from numba.core.types import float64, int64, string, boolean, DictType, UniTuple
    from numba.experimental import jitclass

    node_spec = [
        ('id',int64),
        ('name',string),
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

    def __init__(self, id: int, name: str) -> None:
        """
        Initialize a Node.

        Parameters:
        -------
        id (int): Unique identifier for the node.
        name (str): Name of the node.
        """
        self.id = int(id)
        self.name = str(name)

    def load_datafile(self, datafiles: Dict[str, DataFile]) -> None:
        """
        Load demand data for this node from a set of DataFiles.

        Demand traces should have units of MW/interval.

        Parameters:
        -------
        datafiles (Dict[str, DataFile]): A dictionary of DataFile objects.
        """

        for key, datafile in datafiles.items():
            match datafile.type:
                case 'demand':
                    self.demand_data = np.array(datafile.data[self.name], dtype=np.float64)
                    break
                case _:
                    continue
        return None

    def unload_datafile(self) -> None:   
        """Unload any attached data to free memory."""    
        self.demand_data = None

    def __repr__(self):
        return f"<Node object [{self.id}]{self.name}>"

if JIT_ENABLED:
    line_spec = [
        ('id', int64),
        ('name', string),
        ('length', float64),
        ('node_start', Node.class_type.instance_type),
        ('node_end', Node.class_type.instance_type),
        ('loss_factor', float64),
        ('max_build', float64),
        ('min_build', float64),
        ('capacity', float64),
        ('unit_type', string),
        ('near_optimum_check', bool_),
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

    def __init__(self, id: int, line_dict: Dict[str, str]) -> None:
        """
        Initialize a Line object.

        Parameters:
        -------
        id (int): Unique identifier for the line.
        line_dict (Dict[str, str]): Dictionary with line attributes.
        """
        self.id = id
        self.name = str(line_dict['name'])
        self.length = int(line_dict['length']) # km
        self.node_start = str(line_dict['node_start']) if not is_nan(line_dict['node_start']) else np.nan # Starting node name
        self.node_end = str(line_dict['node_end']) if not is_nan(line_dict['node_end']) else np.nan  # Ending node name
        self.loss_factor = float(line_dict['loss_factor'])  # Transmission losses % per 1000 km
        self.max_build = float(line_dict['max_build'])  # GW/year
        self.min_build = float(line_dict['min_build'])  # GW/year
        self.capacity = float(line_dict['initial_capacity'])  # GW
        self.unit_type = str(line_dict['unit_type'])
        self.near_opt = str(line_dict.get('near_optimum','')).lower() in ('true','1','yes')
        raw_group = line_dict.get('range_group','')
        self.group = self.name if (raw_group is None or (isinstance(raw_group,float) and np.isnan(raw_group)) or str(raw_group).strip()=='') else str(raw_group).strip()
        self.cost = UnitCost(float(line_dict['capex']),
                              float(line_dict['fom']),
                              float(line_dict['vom']),
                              int(line_dict['lifetime']),
                              float(line_dict['discount_rate']),
                              transformer_capex=int(line_dict['transformer_capex']),
                              length=self.length)

    def __repr__(self):
        return f"<Line object [{self.id}]{self.name}>"

if JIT_ENABLED:
    network_spec = [
        ('network',int64[:,:]),
        ('network_mask',boolean[:]),
        ('transmission_mask',boolean[:]),
        ('cache_0_donors',DictType(int64, int64[:, :])),
        ('cache_n_donors',DictType(UniTuple(int64, 2), int64[:, :, :])),
    ]
else:
    network_spec = []

class Network:
    """
    Constructs the network topology for transmission modeling using lines and nodes.
    Provides access to transmission masks, direct connection matrices, and nth-order networks
    required for transmission business rules in the unit commitment problem.
    """

    def __init__(self, lines: Dict[int,Line], nodes: Dict[int,Node], networksteps_max: int) -> None:
        """
        Initialize the Network topology and build all relevant matrices and masks.

        Parameters:
        -------
        lines (Dict[int, Line]): Dictionary of transmission lines.
        nodes (Dict[int, Node]): Dictionary of nodes in the system.
        networksteps_max (int): Maximum number of legs along which transmission can occur.
        """

        lines = self._remove_minor_lines(lines)
        self.topology = self._get_topology(lines, nodes)
        self.node_count = len(nodes)
        self.network, self.network_mask, self.transmission_mask = self._build_base_network()
        self.direct_connections = self._get_direct_connections()
        self.cache_0_donors = self._build_0_donor_cache()
        self.cache_n_donors = self._build_n_donor_cache(networksteps_max)
        
    @staticmethod
    def _remove_minor_lines(lines: Dict[str,Line]) -> Dict[int, Line]:
        """
        Removes minor lines that are used for connecting generator and storage
        units to the transmission network.

        Parameters:
        -------
        lines (Dict[int, Line]): Raw line dictionary containing all lines.

        Returns:
        -------
        Dict[int, Line]: Cleaned line dictionary with minor lines removed.
        """

        return {k: v for k, v in lines.items() if not (is_nan(v.node_start) or is_nan(v.node_end))}

    
    @staticmethod
    def _get_topology(lines: Dict[str,Line], nodes: Dict[str,Node]) -> NDArray[np.int64]:
        """
        Constructs the base topology matrix mapping each line to its start/end nodes.

        Returns:
        -------
        NDArray[np.int64]: Array of shape (num_lines, 2). Each row represents a line. The
                            first column gives the node_start id, the second column gives
                            the node_end id.
        """

        node_ids = {node.name: node.id for node in nodes.values()}
        node_order = {node_id: idx for idx, node_id in enumerate(nodes.keys())}
        line_order = {line.id: idx for idx, line in enumerate(lines.values())}

        topology = np.full((len(lines), 2), -1, dtype=np.int64)
        for key, line in lines.items():
            start = node_order[node_ids[line.node_start]]
            end = node_order[node_ids[line.node_end]]
            topology[line_order[line.id]] = [start, end]
        return topology

    def _build_base_network(self):
        node_range = range(self.node_count)
        mask = np.array([(self.topology == j).sum(axis=1).astype(bool) for j in node_range]).sum(axis=0) == 2
        network = self.topology[mask]
        node_map = {node_id: idx for idx, node_id in enumerate(node_range)}
        network = np.vectorize(node_map.get)(network)

        transmission_mask = np.zeros((self.node_count, network.shape[0]), dtype=bool)
        for line_id, (start, _) in enumerate(network):
            transmission_mask[start, line_id] = True

        return network, mask, transmission_mask

    def _get_direct_connections(self) -> NDArray[np.int64]:
        conn = -1 * np.ones((self.node_count + 1, self.node_count + 1), dtype=np.int64)
        for idx, (i, j) in enumerate(self.network):
            conn[i, j] = idx
            conn[j, i] = idx
        return conn

    def _network_neighbours(self, node: int) -> NDArray[np.int64]:
        return np.where(self.direct_connections[node] != -1)[0]

    def _build_0_donor_cache(self) -> Dict[int, NDArray[np.int64]]:
        cache = {}
        for n in range(self.node_count):
            neighbors = self._network_neighbours(n)
            lines = self.direct_connections[n, neighbors]
            cache[n] = np.stack((neighbors, lines))
        return cache

    def _build_n_donor_cache(self, max_steps: int) -> Dict[tuple[int, int], NDArray[np.int64]]:
        cache = {}
        paths = self.network.copy()

        for step in range(1, max_steps):
            paths = self._nth_order_paths(paths)
            for n in range(self.node_count):
                forward = paths[paths[:, 0] == n]
                reverse = paths[paths[:, -1] == n]
                combined = np.vstack((forward[:, 1:], reverse[:, :-1][:, ::-1]))

                lines = np.empty_like(combined)
                for i in range(combined.shape[0]):
                    lines[i, 0] = self.direct_connections[n, combined[i, 0]]
                    for j in range(1, combined.shape[1]):
                        lines[i, j] = self.direct_connections[combined[i, j - 1], combined[i, j]]

                cache[(n, step)] = np.dstack((combined, lines)).T
        return cache

    def _nth_order_paths(self, paths: NDArray[np.int64]) -> NDArray[np.int64]:
        candidates = []
        for path in paths:
            new_paths = self._extend_path(path)
            candidates.extend(new_paths)

        deduped = self._deduplicate_paths(np.array(candidates, dtype=np.int64))
        return deduped

    def _extend_path(self, path: NDArray[np.int64]) -> list[NDArray[np.int64]]:
        start_neighbors = [n for n in self._network_neighbours(path[0]) if n not in path]
        end_neighbors = [n for n in self._network_neighbours(path[-1]) if n not in path]
        new_paths = []

        for n in start_neighbors:
            new_paths.append(np.insert(path, 0, n))
        for n in end_neighbors:
            new_paths.append(np.append(path, n))

        return new_paths

    def _deduplicate_paths(self, paths: NDArray[np.int64]) -> NDArray[np.int64]:
        def canonical(row):
            return row if tuple(row) < tuple(row[::-1]) else row[::-1]

        canonical_paths = np.array([canonical(row) for row in paths])
        _, idx = np.unique(canonical_paths, axis=0, return_index=True)
        return canonical_paths[np.sort(idx)]
    
@njit
def get_transmission_flows_t2(solution, 
                             Fillt: NDArray[np.float64], 
                             Surplust: NDArray[np.float64], 
                             Importt: NDArray[np.float64], 
                             Exportt: NDArray[np.float64]
                             ) -> NDArray[np.float64]:
    
    # The primary connections are simpler (and faster) to model than the general
    #   nthary connection
    # Since many if not most calls of this function only require primary transmission
    #   I have split it out from general nthary transmission to improve speed
    _transmission = np.zeros(solution.nodes, np.float64)
    leg = 0
    # loop through nodes with deficits
    for n in range(solution.nodes):
        if Fillt[n] < 1e-6:
            continue
        # appropriate slice of network array
        # pdonors is equivalent to donors later on but has different ndim so needs to
        #   be a different variable name for static typing
        pdonors, pdonor_lines = solution.cache_0_donors[n]
        _usage = 0.0 # badly named by avoids creating more variables
        for d in pdonors: 
            _usage += Surplust[d]

        if _usage < 1e-6:
            # continue if no surplus to be traded
            continue

        for d, l in zip(pdonors, pdonor_lines):
            _usage = 0.0
            for m in range(solution.nodes):
                _usage += Importt[m, l]
            # maximum exportable
            _transmission[d] = min(
                Surplust[d],  # power resource constraint
                solution.GHvi[l] - _usage, # line capacity constraint
            )  

        # scale down to fill requirement
        _usage = 0.0
        for m in range(solution.nodes):
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
            _capacity = np.zeros(solution.nhvi, np.float64)
            # loop through secondary, tertiary, ..., nthary connections
            for leg in range(1, solution.networksteps):
                # loop through nodes with deficits
                for n in range(solution.nodes):
                    if Fillt[n] < 1e-6:
                        continue
                    
                    donors, donor_lines = solution.cache_n_donors[(n, leg)]

                    if donors.shape[1] == 0:
                        break  # break if no valid donors
                        
                    _usage = 0.0 # badly named variable but avoids extra variables
                    for d in donors[-1]:
                        _usage += Surplust[d]

                    if _usage < 1e-6:
                        continue

                    _capacity[:] = solution.GHvi - array_sum_2d_axis0(Importt)
                    for d, dl in zip(donors[-1], donor_lines.T): # print(d,dl)
                        # power use of each line, clipped to maximum capacity of lowest leg
                        _import[d, dl] = min(array_min(_capacity[dl]), Surplust[d])
                    
                    for l in range(solution.nhvi):
                        # total usage of the line across all import paths
                        _usage=0.0
                        for m in range(solution.nodes):
                            _usage += _import[m, l]
                        # if usage exceeds capacity
                        if _usage > _capacity[l]:
                            # unclear why this raises zero division error from time to time
                            _scale = zero_safe_division(_capacity[l], _usage)
                            for m in range(solution.nodes):
                                # clip all legs
                                if _import[m, l] > 1e-6:
                                    for o in range(solution.nhvi):
                                        _import[m, o] *= _scale
                        
                    # intermediate calculation array
                    _transmission = array_max_2d_axis1(_import)
                    
                    # scale down to fill requirement
                    _usage = 0.0
                    for m in range(solution.nodes):
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

@njit
def get_transmission_flows_t(Fillt: NDArray[np.float64], 
                             Surplust: NDArray[np.float64], 
                             Hcapacity: NDArray[np.float64], 
                             network: NDArray[np.int64], 
                             networksteps: int, 
                             Importt: NDArray[np.float64], 
                             Exportt: NDArray[np.float64]
                             ) -> NDArray[np.float64]:
    """
    Compute the transmission flows between nodes over time, including primary and nth-order flows.

    Parameters:
    -------
    Fillt (NDArray[np.float64]): 1D array of positive demand (e.g., netload or remaining charging capacity) to fill at each node 
                                    for time interval t [GW].
    Surplust (NDArray[np.float64]): Excess generation available at each node for time interval t [GW].
    Hcapacity (NDArray[np.float64]): Rated capacity for each line [GW].
    network (NDArray[np.int64]): 4D matrix of transmission path routing between nodes.
    networksteps (int): Number of valid network "hops" (1st order, 2nd order, etc.). For example, networksteps of 2 will allow 
                            second-order neighbouring nodes to transmit between each other.
    Importt (NDArray[np.float64]): Matrix of current imports per line and node for time interval t [GW].
    Exportt (NDArray[np.float64]): Matrix of current exports per line and node for time interval t [GW].

    Returns:
    -------
    NDArray[np.float64]: Updated combined import/export transmission flow matrix.
    """

    # The primary connections are simpler (and faster) to model than the general
    #   nthary connection
    # Since many if not most calls of this function only require primary transmission
    #   I have split it out from general nthary transmission to improve speed
    if network.size == 0:
        return Importt+Exportt

    for n in np.where(Fillt>0)[0]:
        pdonors = network[:, n, 0, :]
        valid_mask = pdonors[0] != -1
        pdonors, pdonor_lines = pdonors[0, valid_mask], pdonors[1, valid_mask]
  
        if Surplust[pdonors].sum() == 0:
            continue
  
        _transmission = np.zeros_like(Fillt)
        _transmission[pdonors] = Surplust[pdonors]
        _transmission[pdonors] = np.minimum(_transmission[pdonors], Hcapacity[pdonor_lines]-Importt[pdonor_lines,:].sum(axis=1))
        
        _transmission /= max(1, _transmission.sum()/Fillt[n])
        
        for d, l in zip(pdonors, pdonor_lines):#  print(d,l)
            Importt[l, n] += _transmission[d]
            Exportt[l, d] -= _transmission[d]
            
        Fillt[n] -= _transmission.sum()
        Surplust -= _transmission                

    # Continue with nthary transmission 
    # Note: This code block works for primary transmission too, but is slower
    if Surplust.sum() > 0 and Fillt.sum() > 0:
        for leg in range(1, networksteps):
            for n in np.where(Fillt>0)[0]:
                donors = network[:, n, TRIANGULAR[leg]:TRIANGULAR[leg+1], :]
                donors, donor_lines = donors[0, :, :], donors[1, :, :]
      
                valid_mask = donors[-1] != -1
                if np.prod(~valid_mask):
                    break
                donor_lines = donor_lines[:, valid_mask]
                donors = donors[:, valid_mask]
                if Surplust[donors[-1]].sum() == 0:
                    continue
      
                ndonors = valid_mask.sum()
                donors = np.concatenate((n*np.ones((1, ndonors), np.int64), donors))
                
                _import = np.zeros_like(Importt)
                for d, dl in zip(donors[-1], donor_lines.T): #print(d,dl)
                    _import[dl, d] = Surplust[d]
                
                hostingcapacity = (Hcapacity-Importt.sum(axis=1))
                zmask = hostingcapacity > 0
                _import[zmask] /= np.atleast_2d(np.maximum(1, _import.sum(axis=1)/hostingcapacity)).T[zmask]
                _import[~zmask]*=-1
                _transmission = _import.sum(axis=0)
                for _row in _import:
                    zmask = _row!=0
                    _transmission[zmask] = np.minimum(_row, _transmission)[zmask]
                _transmission=np.maximum(0, _transmission)
                _transmission /= max(1, _transmission.sum()/Fillt[n])
                
                for nd, d, dl in zip(range(ndonors), donors[-1], donor_lines.T):
                    for step, l in enumerate(dl): 
                        Importt[l, donors[step, nd]] += _transmission[d]
                        Exportt[l, donors[step+1, nd]] -= _transmission[d]
                Fillt[n] -= _transmission.sum()
                Surplust -= _transmission                
                
                if Surplust.sum() == 0 or Fillt.sum() == 0:
                    break
                
            if Surplust.sum() == 0 or Fillt.sum() == 0:
                break
        
    return Importt+Exportt