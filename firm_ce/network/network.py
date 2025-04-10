from typing import Dict
import numpy as np

from firm_ce.components import Line, Node
from firm_ce.constants import TRIANGULAR, JIT_ENABLED

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

class Network:
    def __init__(self, lines: Dict[int,Line], nodes: Dict[int,Node]) -> None:
        self.topology = self._get_topology(lines, nodes)
        self.node_count = len(nodes)
        self.transmission_mask = self._get_transmission_mask()
        self.direct_connections = self._get_direct_connections()
        self.topologies_nd = self._get_topologies_nd()
        self.max_connections = max([self.count_lines(topology) for topology in self.topologies_nd])
        self.network = self._get_network()
        self.networksteps = self._get_network_steps()

        self.direct_connections = self.direct_connections[:-1, :-1]

    def _get_network_steps(self):
        return np.where(TRIANGULAR == self.network.shape[2])[0][0]

    def _get_topology(self, lines: Dict[str,Line], nodes: Dict[str,Node]) -> np.ndarray:
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

        for key in lines:
            line = lines[key]
            line_id = line.id
            start_node_id = node_names[line.node_start]
            end_node_id = node_names[line.node_end]
            topology[line_order[line_id]] = [node_order[start_node_id], node_order[end_node_id]]

        return topology
    
    def _get_transmission_mask(self) -> np.ndarray:
        transmission_mask = np.zeros((self.node_count, len(self.topology)), dtype=np.bool_)
        for n, row in enumerate(self.topology):
            transmission_mask[row[0], n] = True
        return transmission_mask
    
    def _get_direct_connections(self) -> np.ndarray:
        direct_connections = np.full((self.node_count+1, self.node_count+1), -1, dtype=np.int64)
        for n, row in enumerate(self.topology):
            i, j = row
            direct_connections[i, j] = n
            direct_connections[j,i] = n
        return direct_connections
    
    def network_neighbours(self, n):
        isn_mask = np.isin(self.topology, n)
        hasn_mask = isn_mask.sum(axis=1).astype(bool)
        joins_n = self.topology[hasn_mask][~isn_mask[hasn_mask]]
        return joins_n
    
    def nthary_network(self, network_1):
        """primary, secondary, tertiary, ..., nthary"""
        """supply n-1thary to generate nthary etc."""
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
    
    def _get_topologies_nd(self):
        topologies_nd = [self.topology]

        while True:
            n = self.nthary_network(topologies_nd[-1])
            if n.size > 0:
                topologies_nd.append(n)
            else: 
                break
        return topologies_nd

    def count_lines(self, network):
        _, counts = np.unique(network[:, np.array([0,-1])], return_counts=True)
        if counts.size > 0:
            return counts.max()
        return 0
    
    def _get_network(self):
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
def get_transmission_flows_t(Fillt, Surplust, Hcapacity, network, networksteps, Importt, Exportt):
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