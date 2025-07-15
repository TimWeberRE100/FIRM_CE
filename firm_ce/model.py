from typing import Dict
import time
from datetime import datetime

from firm_ce.optimisation.statistics import generate_result_files
from firm_ce.io.validate import ModelData

class ModelConfig:
    def __init__(self, config_dict: Dict[str, str]) -> None:
        config_dict = { item['name']: item['value'] for item in config_dict.values() }
        self.type = config_dict['type']
        self.iterations = int(config_dict['iterations'])
        self.population = int(config_dict['population'])
        self.mutation = float(config_dict['mutation'])
        self.recombination = float(config_dict['recombination'])
        self.global_optimal_lcoe = float(config_dict.get('global_optimal_lcoe', 0.0))
        self.near_optimal_tol = float(config_dict.get('near_optimal_tol', 0.0))
        self.midpoint_count = int(config_dict.get('midpoint_count', 0))

class Model:
    def __init__(self) -> None:        
        model_data = ModelData()
        if not model_data.validate():
            exit()

        self.config = ModelConfig(model_data.config)
        self.datafile_filenames_dict = model_data.datafiles
        self.scenarios = {
            model_data.scenarios[scenario_idx].get('scenario_name'): Scenario(model_data,scenario_idx) for scenario_idx in model_data.scenarios 
        }

    def solve(self):
        for scenario in self.scenarios.values(): 
            start_time = time.time()
            start_time_str = datetime.fromtimestamp(start_time).strftime('%d/%m/%Y %H:%M:%S')
            scenario.logger.info(f'Started scenario {scenario.name} at {start_time_str}.')

            scenario.load_datafiles(self.datafile_filenames_dict)     
            datafile_loadtime = time.time()   
            datafile_loadtime_str = datetime.fromtimestamp(datafile_loadtime).strftime('%d/%m/%Y %H:%M:%S')
            scenario.logger.info(f'Datafiles loaded at {datafile_loadtime_str} ({datafile_loadtime - start_time:.4f} seconds).')

            de_result = scenario.solve(self.config)
            solve_time = time.time()   
            solve_time_str = datetime.fromtimestamp(solve_time).strftime('%d/%m/%Y %H:%M:%S')
            scenario.logger.info(f'Optimisation completed at {solve_time_str} ({(solve_time - datafile_loadtime)/(60*60):.4f} hours).')
            
            if self.config.type == 'single_time':
                generate_result_files(de_result.x, scenario, self.config)
                results_time = time.time() 
                results_time_str = datetime.fromtimestamp(results_time).strftime('%d/%m/%Y %H:%M:%S')
                scenario.logger.info(f'Results saved at {results_time_str} ({results_time - solve_time:.4f} seconds).')

            scenario.unload_datafiles()
            end_time = time.time()
            end_time_str = datetime.fromtimestamp(end_time).strftime('%d/%m/%Y %H:%M:%S')
            scenario.logger.info(f'Scenario completed at {end_time_str} (Total time taken: {(end_time - start_time)/(60*60):.4f} hours).')
   