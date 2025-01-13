from typing import Dict
from firm_ce.file_manager import DataFile 
from firm_ce.components.costs import UnitCost

class Node:
    def __init__(self, id: int, name: str, datafiles: Dict[str, DataFile]) -> None:
        self.id = int(id)
        self.name = str(name)

        for key in datafiles:
            if datafiles[key].type != 'demand':
                continue
            self.demand_data = list(datafiles[key].data[self.name])

    def __repr__(self):
        return f"<Node object [{self.id}]{self.name}>"

class Line:
    def __init__(self, id: int, line_dict: Dict[str, str]) -> None:
        self.id = id
        self.name = str(line_dict['name'])
        self.length = int(line_dict['length']) # km
        self.node_start = str(line_dict['node_start'])  # Starting node name
        self.node_end = str(line_dict['node_end'])  # Ending node name
        self.loss_factor = float(line_dict['loss_factor'])  # Transmission losses % per 1000 km

        self.cost = UnitCost(int(line_dict['capex']),
                              float(line_dict['fom']),
                              float(line_dict['vom']),
                              int(line_dict['lifetime']),
                              float(line_dict['discount_rate']),
                              int(line_dict['transformer_capex']),
                              self.length)

    def __repr__(self):
        return f"<Line object [{self.id}]{self.name}>"