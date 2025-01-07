from typing import Dict, List

from firm_ce.file_manager import import_csv_data, DataFile 
from firm_ce.components import Generator, Storage, Line, Node
from firm_ce.optimisation import Constraint, Solver
from firm_ce.network import Network

class ModelData:
    def __init__(self) -> None:
        objects = import_csv_data()

        self.scenarios = objects.get('scenarios')
        self.generators = objects.get('generators')
        self.lines = objects.get('lines')
        self.storages = objects.get('storages')
        self.constraints = objects.get('constraints')
        self.datafiles = objects.get('datafiles')
        self.config = objects.get('config')

class Scenario:
    def __init__(self, model_data: ModelData, scenario_id: int) -> None:
        scenario_data = model_data.scenarios.get(scenario_id) 

        self.id = scenario_id
        self.name = scenario_data.get('scenario_name', '')

        datafiles = self._get_datafiles(model_data.datafiles)

        self.resolution = float(scenario_data.get('resolution', 0.0))
        self.first_year = int(scenario_data.get('firstyear', 0))
        self.final_year = int(scenario_data.get('finalyear', 0))
        self.nodes = self._get_nodes(scenario_data.get('nodes', ''), datafiles)
        self.lines = self._get_lines(model_data.lines)
        self.constraints = self._get_constraints(model_data.constraints)
        self.generators = self._get_generators(model_data.generators, datafiles)
        self.storages = self._get_storages(model_data.storages)
        self.config = self._get_config(model_data.config)
        self.type = scenario_data.get('type', '')
        self.network = Network(self.lines, self.nodes)
        self.x = self._get_decision_x()

    def __repr__(self):
        return f"<Scenario object [{self.id}]{self.name}>"

    @staticmethod
    def _parse_comma_separated(value: str) -> List[str]:
        """Parse a comma-separated string into a clean list of strings."""
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _get_nodes(self, node_names: str, datafiles: Dict[str, DataFile]) -> Dict[str,Node]:
        node_names = self._parse_comma_separated(node_names)
        return {node_names[idx]: Node(idx,node_names[idx], datafiles) for idx in range(len(node_names))}

    def _get_lines(self, all_lines: Dict[str,Dict[str,str]]) -> Dict[str,Line]:
        """Extract line names from scenario data."""
        return {all_lines[idx]['name']: Line(idx, all_lines[idx]) for idx in all_lines if self.name in self._parse_comma_separated(all_lines[idx]['scenarios'])}

    def _get_generators(self, all_generators: Dict[str,Dict[str,str]], datafiles: Dict[str, DataFile]) -> Dict[str,Generator]:
        """Filter or prepare generator data specific to this scenario."""
        return {all_generators[idx]['name'] : Generator(idx, all_generators[idx], datafiles) for idx in all_generators if self.name in self._parse_comma_separated(all_generators[idx]['scenarios'])}
    
    def _get_storages(self, all_storages: Dict[str,Dict[str,str]]) -> Dict[str,Storage]:
        """Filter or prepare storage data specific to this scenario."""
        return {all_storages[idx]['name'] : Storage(idx, all_storages[idx]) for idx in all_storages if self.name in self._parse_comma_separated(all_storages[idx]['scenarios'])}
    
    def _get_constraints(self, all_constraints: Dict[str,Dict[str,str]]) -> Dict[str,Constraint]:
        """Filter or prepare constraint data specific to this scenario."""
        return {all_constraints[idx]['name'] : Constraint(idx, all_constraints[idx]) for idx in all_constraints if self.name in self._parse_comma_separated(all_constraints[idx]['scenarios'])}

    def _get_datafiles(self, all_datafiles: Dict[str,Dict[str,str]]) -> Dict[str,Constraint]:
        """Filter or prepare datafiles specific to this scenario."""
        return {all_datafiles[idx]['filename']: DataFile(all_datafiles[idx]['filename'],all_datafiles[idx]['datafile_type']) for idx in all_datafiles if self.name in self._parse_comma_separated(all_datafiles[idx]['scenarios'])}

    def _get_config(self, all_configs: Dict[str,Dict[str,str]]) -> Dict[str,float]:
        """Filter or prepare optimiser config specific to this scenario."""
        return {all_configs[idx]['name']: float(all_configs[idx]['value']) for idx in all_configs if self.name in self._parse_comma_separated(all_configs[idx]['scenarios'])}

    def _get_decision_x(self):
        pass

    def solve(self):
        solver = Solver(self.type,self.config)
        solver.solve()

class Model:
    def __init__(self):
        model_data = ModelData()

        self.scenarios = {
            model_data.scenarios[scenario_idx].get('scenario_name'): Scenario(model_data,scenario_idx) for scenario_idx in model_data.scenarios 
        }

    def solve(self):
        for scenario in self.scenarios.values():
            scenario.solve()
