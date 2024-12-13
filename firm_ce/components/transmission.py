from typing import Dict

class Node:
    def __init__(self, id: int, name: str) -> None:
        self.id = int(id)
        self.name = str(name)

class Line:
    def __init__(self, line_dict: Dict[str]) -> None:
        self.id = int(line_dict['id'])
        self.name = str(line_dict['name'])
        self.length = int(line_dict['length']) # km
        self.capex = float(line_dict['capex'])  # $
        self.fom = float(line_dict['fom'])  # $/kW-year
        self.vom = float(line_dict['vom'])  # $/kWh
        self.lifetime = int(line_dict['lifetime'])  # years
        self.discount_rate = float(line_dict['discount_rate']) #[0,1]
        self.node_start = str(line_dict['node_start'])  # Starting node name
        self.node_end = str(line_dict['node_end'])  # Ending node name
        self.loss_factor = float(line_dict['loss_factor'])  # Transmission losses % per 1000 km