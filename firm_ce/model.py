from typing import Dict, List

from firm_ce.file_manager import import_csv_data, DataFile 
from firm_ce.components import Generator, Storage, Line, Node
from firm_ce.optimisation import Solver
from firm_ce.network import Network
from firm_ce.network.frequency import get_frequencies

class ModelData:
    def __init__(self) -> None:
        objects = import_csv_data()

        self.scenarios = objects.get('scenarios')
        self.generators = objects.get('generators')
        self.lines = objects.get('lines')
        self.storages = objects.get('storages')
        self.datafiles = objects.get('datafiles')
        self.config = objects.get('config')
        self.settings = objects.get('settings')

class Scenario:
    def __init__(self, model_data: ModelData, scenario_id: int) -> None:
        scenario_data = model_data.scenarios.get(scenario_id) 

        self.id = scenario_id
        self.name = scenario_data.get('scenario_name', '')

        datafiles = self._get_datafiles(model_data.datafiles)

        self.resolution = float(scenario_data.get('resolution', 0.0))
        self.allowance = float(scenario_data.get('allowance', 0.0))
        self.first_year = int(scenario_data.get('firstyear', 0))
        self.final_year = int(scenario_data.get('finalyear', 0))
        self.nodes = self._get_nodes(scenario_data.get('nodes', ''), datafiles)
        self.lines = self._get_lines(model_data.lines)
        self.generators = self._get_generators(model_data.generators, datafiles)
        self.storages = self._get_storages(model_data.storages)
        self.type = scenario_data.get('type', '')
        self.network = Network(self.lines, self.nodes)

        self.intervals = len(self.nodes[0].demand_data)
        self.node_names = {self.nodes[idx].name : self.nodes[idx].id for idx in self.nodes}
        self.nodes_with_balancing = set([self.node_names[self.storages[idx].node] for idx in self.storages] 
                                        + [self.node_names[self.generators[idx].node] for idx in self.generators if self.generators[idx].unit_type == 'flexible'])
        self.max_frequency = max(get_frequencies(self.intervals, self.resolution))

    def __repr__(self):
        return f"<Scenario object [{self.id}]{self.name}>"

    @staticmethod
    def _parse_comma_separated(value: str) -> List[str]:
        """Parse a comma-separated string into a clean list of strings."""
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _get_nodes(self, node_names: str, datafiles: Dict[str, DataFile]) -> Dict[str,Node]:
        node_names = self._parse_comma_separated(node_names)
        return {idx: Node(idx,node_names[idx], datafiles) for idx in range(len(node_names))}

    def _get_lines(self, all_lines: Dict[str,Dict[str,str]]) -> Dict[str,Line]:
        """Extract line names from scenario data."""
        return {idx: Line(idx, all_lines[idx]) for idx in all_lines if self.name in self._parse_comma_separated(all_lines[idx]['scenarios'])}

    def _get_generators(self, all_generators: Dict[str,Dict[str,str]], datafiles: Dict[str, DataFile]) -> Dict[str,Generator]:
        """Filter or prepare generator data specific to this scenario."""
        return {idx: Generator(idx, all_generators[idx], datafiles) for idx in all_generators if self.name in self._parse_comma_separated(all_generators[idx]['scenarios'])}
    
    def _get_storages(self, all_storages: Dict[str,Dict[str,str]]) -> Dict[str,Storage]:
        """Filter or prepare storage data specific to this scenario."""
        return {idx: Storage(idx, all_storages[idx]) for idx in all_storages if self.name in self._parse_comma_separated(all_storages[idx]['scenarios'])}
    
    def _get_datafiles(self, all_datafiles: Dict[str,Dict[str,str]]) -> Dict[str,DataFile]:
        """Filter or prepare datafiles specific to this scenario."""
        return {all_datafiles[idx]['filename']: DataFile(all_datafiles[idx]['filename'],all_datafiles[idx]['datafile_type']) for idx in all_datafiles if self.name in self._parse_comma_separated(all_datafiles[idx]['scenarios'])}

    def solve(self, config):
        solver = Solver(config, self)
        solver.evaluate()
        return solver.solution

class ModelSettings:
    def __init__(self, settings_dict: Dict[str, str]) -> None:
        self.generator_unit_types = settings_dict['generator_unit_types']
        self.storage_unit_types = settings_dict['storage_unit_types']
        self.line_unit_types = settings_dict['line_unit_types']

class ModelConfig:
    def __init__(self, config_dict: Dict[str, str], settings_dict: Dict[str, str]) -> None:
        config_dict = { item['name']: item['value'] for item in config_dict.values() }
        self.type = config_dict['type']
        self.iterations = int(config_dict['iterations'])
        self.population = int(config_dict['population'])
        self.mutation = float(config_dict['mutation'])
        self.recombination = float(config_dict['recombination'])
        self.settings = ModelSettings(settings_dict)

class Model:
    def __init__(self) -> None:
        model_data = ModelData()

        self.config = ModelConfig(model_data.config, model_data.settings)
        self.scenarios = {
            model_data.scenarios[scenario_idx].get('scenario_name'): Scenario(model_data,scenario_idx) for scenario_idx in model_data.scenarios 
        }
        self.results = {}

    def solve(self):
        for scenario in self.scenarios.values():
            
            self.results[scenario.name] = scenario.solve(self.config)
            exit() ####### DEBUG