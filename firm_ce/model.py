from typing import Dict, List

from firm_ce.file_manager import import_csv_data 
from firm_ce.components import Generator, Storage, Line, Node
from firm_ce.optimisation import Constraint
from firm_ce.network import Network

class ModelData:
    def __init__(self) -> None:
        objects = import_csv_data()

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

    def __repr__(self):
        return f"{self.id}: {self.name}"

    @staticmethod
    def _parse_comma_separated(value: str) -> List[str]:
        """Parse a comma-separated string into a clean list of strings."""
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _get_nodes(self, node_names: str) -> Dict[str,Node]:
        node_names = self._parse_comma_separated(node_names)
        return {node_names[idx]: Node(idx,node_names[idx]) for idx in range(len(node_names))}

    def _get_lines(self, all_lines: Dict[str,Dict[str,str]]) -> Dict[str,Line]:
        """Extract line names from scenario data."""
        return {all_lines[line_idx]['name']: Line(line_idx, all_lines[line_idx]) for line_idx in all_lines if self.name in self._parse_comma_separated(all_lines[line_idx]['scenarios'])}

    def _get_generators(self, all_generators: Dict[str,Dict[str,str]]) -> Dict[str,Generator]:
        """Filter or prepare generator data specific to this scenario."""
        return {all_generators[gen_idx]['name'] : Generator(gen_idx, all_generators[gen_idx]) for gen_idx in all_generators if self.name in self._parse_comma_separated(all_generators[gen_idx]['scenarios'])}
    
    def _get_storages(self, all_storages: Dict[str,Dict[str,str]]) -> Dict[str,Storage]:
        """Filter or prepare storage data specific to this scenario."""
        return {all_storages[storage_idx]['name'] : Storage(storage_idx, all_storages[storage_idx]) for storage_idx in all_storages if self.name in self._parse_comma_separated(all_storages[storage_idx]['scenarios'])}
    
    def _get_constraints(self, all_constraints: Dict[str,Dict[str,str]]) -> Dict[str,Constraint]:
        """Filter or prepare constraint data specific to this scenario."""
        return {all_constraints[constraint_idx]['name'] : Constraint(constraint_idx, all_constraints[constraint_idx]) for constraint_idx in all_constraints if self.name in self._parse_comma_separated(all_constraints[constraint_idx]['scenarios'])}

class Model:
    def __init__(self):
        model_data = ModelData()

        #self.config = model_data.config ### Add config settings for entire model
        self.scenarios = {
            model_data.scenarios[scenario_idx].get('scenario_name'): Scenario(model_data,scenario_idx) for scenario_idx in model_data.scenarios 
        }

