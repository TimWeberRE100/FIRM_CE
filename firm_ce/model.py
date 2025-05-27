from typing import Dict, List
import numpy as np
import gc

from firm_ce.file_manager import import_csv_data, import_datafiles, DataFile
from firm_ce.components import Generator, Storage, Line, Node, Fuel
from firm_ce.optimisation import Solver
from firm_ce.network import Network
from firm_ce.optimisation.statistics import generate_result_files

class ModelData:
    def __init__(self) -> None:
        objects = import_csv_data()

        self.scenarios = objects.get('scenarios')
        self.generators = objects.get('generators')
        self.fuels = objects.get('fuels')
        self.lines = objects.get('lines')
        self.storages = objects.get('storages')
        self.config = objects.get('config')
        self.x0s = objects.get('initial_guess')
        self.settings = objects.get('settings')

class Scenario:
    def __init__(self, model_data: ModelData, scenario_id: int) -> None:
        scenario_data = model_data.scenarios.get(scenario_id) 

        self.id = scenario_id
        self.name = scenario_data.get('scenario_name', '')

        self.x0 = self._get_x0(model_data.x0s)

        self.resolution = float(scenario_data.get('resolution', 0.0))
        self.allowance = float(scenario_data.get('allowance', 0.0))
        self.first_year = int(scenario_data.get('firstyear', 0))
        self.final_year = int(scenario_data.get('finalyear', 0))
        self.nodes = self._get_nodes(scenario_data.get('nodes', ''))
        self.lines = self._get_lines(model_data.lines)
        self.fuels = self._get_fuels(model_data.fuels)
        self.generators = self._get_generators(model_data.generators, self.fuels, self.lines)
        self.storages = self._get_storages(model_data.storages, self.lines)
        self.type = scenario_data.get('type', '')
        self.network = Network(self.lines, self.nodes)
        self.intervals = 0

        self.node_names = {self.nodes[idx].name : self.nodes[idx].id for idx in self.nodes}
        self.nodes_with_balancing = set([self.node_names[self.storages[idx].node] for idx in self.storages] 
                                        + [self.node_names[self.generators[idx].node] for idx in self.generators if self.generators[idx].unit_type == 'flexible'])
    
    def __repr__(self):
        return f"<Scenario object [{self.id}]{self.name}>"
    
    def load_datafiles(self):
        all_datafiles = import_datafiles()

        datafiles = self._get_datafiles(all_datafiles)

        for g in self.generators.values():
            g.load_datafile(datafiles)
        
        for n in self.nodes.values():
            n.load_datafile(datafiles)

        self.intervals = len(self.nodes[0].demand_data)

    def unload_datafiles(self):
        for g in self.generators.values():
            g.unload_datafile()
        
        for n in self.nodes.values():
            n.unload_datafile()

        self.intervals = 0

        gc.collect()

    @staticmethod
    def _parse_comma_separated(value: str) -> List[str]:
        """Parse a comma-separated string into a clean list of strings."""
        return [item.strip() for item in value.split(',') if item.strip()]
    
    @staticmethod
    def _get_generator_fuels(all_generators: Dict[str,Dict[str,str]], fuel_dict: Dict[str,Fuel]) -> Dict[str,Fuel]:
        fuel_name_map = {fuel_dict[idx].name: fuel_dict[idx] for idx in fuel_dict}
        return [fuel_name_map[all_generators[g]['fuel']] for g in all_generators if all_generators[g]['fuel'] in fuel_name_map]
    
    @staticmethod
    def _get_component_lines(all_components: Dict[str,Dict[str,str]], line_dict: Dict[str,Line]) -> Dict[str,Line]:
        line_name_map = {line_dict[idx].name: line_dict[idx] for idx in line_dict}
        result = []
        for g in all_components:
            if all_components[g]['line'] in line_name_map:
                result.append(line_name_map[all_components[g]['line']])
            else:
                result.append(None)
        return result
    
    def _get_nodes(self, node_names: str) -> Dict[str,Node]:
        node_names = self._parse_comma_separated(node_names)
        return {idx: Node(idx,node_names[idx]) for idx in range(len(node_names))}

    def _get_lines(self, all_lines: Dict[str,Dict[str,str]]) -> Dict[str,Line]:
        """Extract line names from scenario data."""
        return {idx: Line(idx, all_lines[idx]) for idx in all_lines if self.name in self._parse_comma_separated(all_lines[idx]['scenarios'])}

    def _get_generators(self, all_generators: Dict[str,Dict[str,str]], fuel_dict: Dict[str,Fuel], line_dict: Dict[str,Line]) -> Dict[str,Generator]:
        """Filter or prepare generator data specific to this scenario."""
        fuels = self._get_generator_fuels(all_generators, fuel_dict)
        lines = self._get_component_lines(all_generators, line_dict)
        return {idx: Generator(idx, all_generators[idx], fuels[idx], lines[idx]) for idx in all_generators if self.name in self._parse_comma_separated(all_generators[idx]['scenarios'])}
    
    def _get_storages(self, all_storages: Dict[str,Dict[str,str]], line_dict: Dict[str,Line]) -> Dict[str,Storage]:
        """Filter or prepare storage data specific to this scenario."""
        lines = self._get_component_lines(all_storages, line_dict)
        return {idx: Storage(idx, all_storages[idx], lines[idx]) for idx in all_storages if self.name in self._parse_comma_separated(all_storages[idx]['scenarios'])}
    
    def _get_fuels(self, all_fuels: Dict[str,Dict[str,str]]) -> Dict[str,Fuel]:
        """Filter or prepare fuel data specific to this scenario."""
        return {idx: Fuel(idx, all_fuels[idx]) for idx in all_fuels if self.name in self._parse_comma_separated(all_fuels[idx]['scenarios'])}
    
    def _get_datafiles(self, all_datafiles: Dict[str,Dict[str,str]]) -> Dict[str,DataFile]:
        """Filter or prepare datafiles specific to this scenario."""
        return {all_datafiles[idx]['filename']: DataFile(all_datafiles[idx]['filename'],all_datafiles[idx]['datafile_type']) for idx in all_datafiles if self.name in self._parse_comma_separated(all_datafiles[idx]['scenarios'])}

    def _get_x0(self, all_x0s: Dict[str,Dict[str,str]]) -> np.ndarray:
        """Get the initial guess corresponding to this scenario."""
        for entry in all_x0s.values():
            if entry.get('scenario') == self.name:
                if (isinstance(entry.get('x_0', ''), float) and np.isnan(entry.get('x_0', ''))):
                    x0_list = []
                else:
                    x0_str = entry.get('x_0', '').strip()    
                    x0_list = [float(x) for x in x0_str.split(',') if x.strip()]
                return np.array(x0_list, dtype=np.float64)
        return None

    def solve(self, config):
        solver = Solver(config, self)
        solver.evaluate()
        return solver.result

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

    def solve(self):
        for scenario in self.scenarios.values(): 
            scenario.load_datafiles()          
            result_x = scenario.solve(self.config)
            generate_result_files(result_x, scenario, self.config)
            scenario.unload_datafiles()
            
