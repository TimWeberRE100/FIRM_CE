from typing import Dict
import time
from datetime import datetime

from firm_ce.system.scenario import Scenario
from firm_ce.optimisation.statistics import Statistics
from firm_ce.system.parameters import ModelConfig
from firm_ce.io.validate import ModelData
from firm_ce.common.constants import DEBUG

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
            
            """ if self.config.type == 'single_time':
                scenario.statistics = Statistics(
                    de_result.x,
                    scenario.static,
                    scenario.fleet,
                    scenario.network,
                    scenario.results_dir,
                    scenario.name,
                    True 
                )
                scenario.statistics.generate_result_files()
                scenario.statistics.write_results()
                if DEBUG:
                    scenario.statistics.dump()
                results_time = time.time() 
                results_time_str = datetime.fromtimestamp(results_time).strftime('%d/%m/%Y %H:%M:%S')
                scenario.logger.info(f'Results saved at {results_time_str} ({results_time - solve_time:.4f} seconds).') """

            scenario.unload_datafiles()
            end_time = time.time()
            end_time_str = datetime.fromtimestamp(end_time).strftime('%d/%m/%Y %H:%M:%S')
            scenario.logger.info(f'Scenario completed at {end_time_str} (Total time taken: {(end_time - start_time)/(60*60):.4f} hours).')
            exit() ###### DEBUG

if __name__ == '__main__':
    model = Model()
    for scenario in model.scenarios.values():
        scenario.load_datafiles()  
        scenario.statistics.generate_result_files()
        scenario.statistics.write_results()
        scenario.unload_datafiles()