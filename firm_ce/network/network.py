from typing import Dict
import numpy as np

from components import Line, Node

TRIANGULAR = np.array([0,1,3,6,10,15,21,28,36])

class Network:
    def __init__(self, lines: Dict[Line], nodes: Dict[Node]) -> None:
        self.topology = self._get_topology(lines, nodes)
        self.node_count = len(nodes)
        self.transmission_mask = self._get_transmission_mask()
        self.direct_connections = self._get_direct_connections()
        self.topologies_nd = self._get_topologies_nd()
        self.max_connections = max([self.count_lines(topology) for topology in self.topologies_nd])
        self.network = self._get_network()

        self.direct_connections = self.direct_connections[:-1, :-1]

    def _get_topology(lines: Dict[Line], nodes: Dict[Node]) -> np.ndarray:
        num_lines = len(lines)
        topology = np.full((num_lines, 2), -1, dtype=np.int64)

        for line in lines.values():
            line_id = line['id']
            start_node_id = nodes[line['node_start']]['id']
            end_node_id = nodes[line['node_end']]['id']
            topology[line_id] = [start_node_id, end_node_id]

        return topology
    
    def _get_transmission_mask(self) -> np.ndarray:
        transmission_mask = np.zeros((MLoad.shape[1], len(self.topology)), dtype=np.bool_)
        for n, row in enumerate(self.topology):
            transmission_mask[row[0], n] = True
        return transmission_mask
    
    def _get_direct_connections(self) -> np.ndarray:
        direct_connections = np.full((self.node_count+1, self.node_count+1), -1, dtype=np.int64)
        for n, row in enumerate(self.topology):
            direct_connections[*row] = n
            direct_connections[*row[::-1]] = n
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

    def count_lines(network):
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
    
