from typing import Dict, List

from file_manager import import_csv_data 
from components import Generator, Storage, Line, Node
from optimisation import Constraint
from network import Network

class ModelData:
    def __init__(self) -> None:
        objects = import_csv_data(self)

        self.scenarios = objects.get('scenarios')
        self.generators = objects.get('generators')
        self.lines = objects.get('lines')
        self.storages = objects.get('storages')
        self.constraints = objects.get('constraints')

class Scenario:
    def __init__(self, model_data: ModelData, scenario_id: int) -> None:
        scenario_data = model_data.scenarios.get(scenario_id) 

        self.id = scenario_id
        self.name = scenario_data.get('scenario_name', '')
        self.resolution = float(scenario_data.get('resolution', 0.0))
        self.first_year = int(scenario_data.get('firstyear', 0))
        self.final_year = int(scenario_data.get('finalyear', 0))
        self.nodes = self._get_nodes(scenario_data.get('nodes', ''))
        self.lines = self._get_lines(model_data.lines)
        self.constraints = self._get_constraints(model_data.constraints)
        self.generators = self._get_generators(model_data.generators)
        self.storages = self._get_storages(model_data.storages)
        self.type = scenario_data.get('type', '')
        self.network = Network(self.lines, self.nodes)

    @staticmethod
    def _parse_comma_separated(value: str) -> List[str]:
        """Parse a comma-separated string into a clean list of strings."""
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _get_nodes(self, node_names: str) -> Dict[Node]:
        node_names = self._parse_comma_separated(node_names)
        return {node_names[idx]: Node(idx,node_names[idx]) for idx in range(len(node_names))}

    def _get_lines(self, all_lines: Dict[Dict[str]]) -> Dict[Line]:
        """Extract line names from scenario data."""
        return {line_dict['name']: Line(line_dict) for line_dict in all_lines if self.name in self._parse_comma_separated(line_dict['scenarios'])}

    def _get_generators(self, all_generators: Dict[Dict[str]]) -> Dict[Generator]:
        """Filter or prepare generator data specific to this scenario."""
        return {gen_dict['name'] : Generator(gen_dict) for gen_dict in all_generators if self.name in self._parse_comma_separated(gen_dict['scenarios'])}
    
    def _get_storages(self, all_storages: Dict[Dict[str]]) -> Dict[Storage]:
        """Filter or prepare storage data specific to this scenario."""
        return {storage_dict['name'] : Storage(storage_dict) for storage_dict in all_storages if self.name in self._parse_comma_separated(storage_dict['scenarios'])}
    
    def _get_constraints(self, all_constraints: Dict[Dict[str]]) -> Dict[Constraint]:
        """Filter or prepare constraint data specific to this scenario."""
        return {constraint_dict['name'] : Constraint(constraint_dict) for constraint_dict in all_constraints if self.name in self._parse_comma_separated(constraint_dict['scenarios'])}

    class Model:
        def __init__(self):
            model_data = ModelData()

            #self.config = model_data.config ### Add config settings for entire model
            self.scenarios = {
                scenario.get('name'): Scenario(model_data,scenario.get('id')) for scenario in model_data.scenarios 
            }

