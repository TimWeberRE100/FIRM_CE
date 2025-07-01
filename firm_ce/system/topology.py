from typing import Dict
import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import TRIANGULAR, JIT_ENABLED
from firm_ce.io.file_manager import DataFile 
from firm_ce.system.costs import UnitCost
from firm_ce.io.validate import is_nan

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

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
        self.demand_data = None

    def load_datafile(self, datafiles: Dict[str, DataFile]) -> None:
        """
        Load demand data for this node from a set of DataFiles.

        Demand traces should have units of MW/interval.

        Parameters:
        -------
        datafiles (Dict[str, DataFile]): A dictionary of DataFile objects.
        """

        for key in datafiles:
            if datafiles[key].type != 'demand':
                continue
            self.demand_data = list(datafiles[key].data[self.name])

    def unload_datafile(self) -> None:   
        """Unload any attached data to free memory."""    
        self.demand_data = None

    def __repr__(self):
        return f"<Node object [{self.id}]{self.name}>"

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
        self.group    = line_dict.get('range_group','')
        self.cost = UnitCost(float(line_dict['capex']),
                              float(line_dict['fom']),
                              float(line_dict['vom']),
                              int(line_dict['lifetime']),
                              float(line_dict['discount_rate']),
                              transformer_capex=int(line_dict['transformer_capex']),
                              length=self.length)

    def __repr__(self):
        return f"<Line object [{self.id}]{self.name}>"
    
class Network:
    """
    Constructs the network topology for transmission modeling using lines and nodes.
    Provides access to transmission masks, direct connection matrices, and nth-order networks
    required for transmission business rules in the unit commitment problem.
    """

    def __init__(self, lines: Dict[int,Line], nodes: Dict[int,Node]) -> None:
        """
        Initialize the Network topology and build all relevant matrices and masks.

        Parameters:
        -------
        lines (Dict[int, Line]): Dictionary of transmission lines.
        nodes (Dict[int, Node]): Dictionary of nodes in the system.
        """

        lines = self._remove_minor_lines(lines)
        self.topology = self._get_topology(lines, nodes)
        self.node_count = len(nodes)
        self.transmission_mask = self._get_transmission_mask()
        self.direct_connections = self._get_direct_connections()
        self.topologies_nd = self._get_topologies_nd()
        self.max_connections = max([self.count_lines(topology) for topology in self.topologies_nd])
        self.network = self._get_network()
        self.networksteps = self._get_network_steps()

        self.direct_connections = self.direct_connections[:-1, :-1]

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

        cleaned_lines = {}
        for key in lines:
            if not (is_nan(lines[key].node_start) or is_nan(lines[key].node_end)):
                cleaned_lines[key] = lines[key]
        return cleaned_lines

    def _get_network_steps(self) -> int:
        """
        Returns the number of network steps (primary, secondary, ...) available.

        Returns:
        -------
        int: Index of the final valid TRIANGULAR number.
        """
        return np.where(TRIANGULAR == self.network.shape[2])[0][0]

    def _get_topology(self, lines: Dict[str,Line], nodes: Dict[str,Node]) -> NDArray[np.int64]:
        """
        Constructs the base topology matrix mapping each line to its start/end nodes.

        Returns:
        -------
        NDArray[np.int64]: Array of shape (num_lines, 2). Each row represents a line. The
                            first column gives the node_start id, the second column gives
                            the node_end id.
        """

        num_lines = len(lines)
        topology = np.full((num_lines, 2), -1, dtype=np.int64)
        node_names = {nodes[idx].name : nodes[idx].id for idx in nodes}
        
        l = 0
        n = 0
        line_order = {}
        node_order = {}
        for key in lines:
            line_order[key] = l
            l += 1
        for key in nodes:
            node_order[key] = n
            n += 1

        l = 0
        n = 0
        line_order = {}
        node_order = {}
        for key in lines:
            line_order[key] = l
            l += 1
        for key in nodes:
            node_order[key] = n
            n += 1

        for key in lines:
            line = lines[key]
            line_id = line.id
            start_node_id = node_names[line.node_start]
            end_node_id = node_names[line.node_end]
            topology[line_order[line_id]] = [node_order[start_node_id], node_order[end_node_id]]

        return topology
    
    def _get_transmission_mask(self) -> NDArray[np.bool_]:
        """
        Generate the transmission mask which is applied to the 3D Transmission matrix
        to select elements necessary for calculating transmission flows over the
        timeseries.

        Returns:
        -------
        NDArray[np.bool_]: Array of shape (1, num_lines, num_nodes).
        """
        transmission_mask = np.zeros((self.node_count, len(self.topology)), dtype=np.bool_)
        for n, row in enumerate(self.topology):
            transmission_mask[row[0], n] = True
        
        return np.atleast_3d(transmission_mask).T
    
    def _get_direct_connections(self) -> NDArray[np.int64]:
        """
        Generate a direct connection matrix of line indices between node pairs.

        Returns:
        -------
        NDArray[np.int64]: Square matrix of shape (node_count + 1, node_count + 1).
        """
        direct_connections = np.full((self.node_count+1, self.node_count+1), -1, dtype=np.int64)
        for n, row in enumerate(self.topology):
            i, j = row
            direct_connections[i, j] = n
            direct_connections[j,i] = n
        return direct_connections
    
    def network_neighbours(self, n: int) -> NDArray[np.int64]:
        """
        Find all node connections that include the given node.

        Parameters:
        -------
        node (int): Node index.

        Returns:
        -------
        NDArray[np.int64]: Array of connected node indices.
        """
        isn_mask = np.isin(self.topology, n)
        hasn_mask = isn_mask.sum(axis=1).astype(bool)
        joins_n = self.topology[hasn_mask][~isn_mask[hasn_mask]]
        return joins_n
    
    def nthary_network(self, network_1: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Generate the next-order network (2nd, 3rd, etc.) from existing paths.

        Parameters:
        -------
        network_1 (NDArray[np.int64]): Existing n-th order network path of shape (paths, steps).

        Returns:
        -------
        NDArray[np.int64]: New n+1 order network with extended paths.
        """
        networkn = -1*np.ones((1,network_1.shape[1]+1), dtype=np.int64)
        for row in network_1:
            _networkn = -1*np.ones((1,network_1.shape[1]+1), dtype=np.int64)
            joins_start = self.network_neighbours(row[0])
            joins_end = self.network_neighbours(row[-1])
            for n in joins_start:
                if n not in row:
                    _networkn = np.vstack((_networkn, np.insert(row, 0, n)))
            for n in joins_end:
                if n not in row:
                    _networkn = np.vstack((_networkn, np.append(row, n)))
            _networkn=_networkn[1:,:]
            dup=[]
            # find rows which are already in network
            for i, r in enumerate(_networkn): 
                for s in networkn:
                    if np.setdiff1d(r, s).size==0:
                        dup.append(i)
            # find duplicated rows within n3
            for i, r in enumerate(_networkn):
                for j, s in enumerate(_networkn):
                    if i==j:
                        continue
                    if np.setdiff1d(r, s).size==0:
                        dup.append(i)
            _networkn = np.delete(_networkn, np.unique(np.array(dup, dtype=np.int64)), axis=0)
            if _networkn.size>0:
                networkn = np.vstack((networkn, _networkn))
        networkn = networkn[1:,:]
        return networkn
    
    def _get_topologies_nd(self) -> list[NDArray[np.int64]]:
        """
        Iteratively build higher-order network topologies.

        Returns:
        -------
        List[NDArray[np.int64]]: List of arrays, each representing a hop level.
        """
        topologies_nd = [self.topology]

        while True:
            n = self.nthary_network(topologies_nd[-1])
            if n.size > 0:
                topologies_nd.append(n)
            else: 
                break
        return topologies_nd

    def count_lines(self, network: NDArray[np.int64]) -> int:
        """
        Count the maximum number of concurrent connections at any node.

        Parameters:
        -------
        network (NDArray[np.int64]): Network array.

        Returns:
        -------
        int: Max number of connections.
        """
        _, counts = np.unique(network[:, np.array([0,-1])], return_counts=True)
        if counts.size > 0:
            return counts.max()
        return 0
    
    def _get_network(self) -> NDArray[np.int64]:
        """
        Build a 4D network array encoding all paths along lines between nodes.

        Returns:
        -------
        NDArray[np.int64]: Shape (2, node_count, TRIANGULAR, max_connections).
        """

        network = -1*np.ones((2, self.node_count, TRIANGULAR[len(self.topologies_nd)], self.max_connections), dtype=np.int64)
        for i, net in enumerate(self.topologies_nd):
            conns = np.zeros(self.node_count, int)
            for j, row in enumerate(net):
                network[0, row[0], TRIANGULAR[i]:TRIANGULAR[i+1], conns[row[0]]] = row[1:]
                network[0, row[-1], TRIANGULAR[i]:TRIANGULAR[i+1], conns[row[-1]]] = row[:-1][::-1]
                conns[row[0]]+=1
                conns[row[-1]]+=1
                
        for i in range(network.shape[1]):
            for j in range(network.shape[2]):
                for k in range(network.shape[3]):
                    if j in TRIANGULAR:
                        start=i
                    else: 
                        start=network[0, i, j-1, k]
                    network[1, i, j, k] = self.direct_connections[start, network[0, i, j, k]]
        return network
    
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