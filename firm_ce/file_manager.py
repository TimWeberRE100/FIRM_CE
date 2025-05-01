from pathlib import Path
import pandas as pd
from typing import List
import numpy as np

class ImportCSV:
    def __init__(self, repository: Path) -> None:
        self.repository = Path(repository)

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def get_data(self, filename: str) -> dict:
        filepath = self.repository.joinpath(filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        return pd.read_csv(filepath, index_col="id").to_dict(orient='index')
    
    def get_scenarios(self) -> dict:
        return self.get_data("scenarios.csv")
    
    def get_generators(self) -> dict:
        return self.get_data("generators.csv")  
    
    def get_fuels(self) -> dict:
        return self.get_data("fuels.csv")  
    
    def get_lines(self) -> dict:
        return self.get_data("lines.csv")
    
    def get_storages(self) -> dict:
        return self.get_data("storages.csv")
    
    def get_datafiles(self) -> dict:
        return self.get_data("datafiles.csv")
    
    def get_config(self) -> dict:
        return self.get_data("config.csv")
    
    def get_initial_guess(self) -> dict:
        return self.get_data("initial_guess.csv")

    def get_setting(self, setting_filename) -> dict:
        return self.get_data(setting_filename)
        
class ImportDatafile:
    def __init__(self, repository: Path, filename: str) -> None:
        self.repository = Path(repository)
        self.filename = filename

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def get_data(self) -> dict:
        filepath = self.repository.joinpath(self.filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        df = pd.read_csv(filepath)
        return df.to_dict(orient='list')

class DataFile:
    def __init__(self, filename, datafile_type):
        self.name = filename
        self.type = datafile_type
        self.data = ImportDatafile("firm_ce/data", filename).get_data()

    def __repr__(self):
        return f"{self.name} ({self.type})"

def import_csv_data() -> dict:
    csv_importer = ImportCSV("firm_ce/config")
    data = {
        'scenarios': csv_importer.get_scenarios(),
        'generators': csv_importer.get_generators(),
        'fuels': csv_importer.get_fuels(),
        'lines': csv_importer.get_lines(),
        'storages': csv_importer.get_storages(),
        'datafiles': csv_importer.get_datafiles(),
        'config': csv_importer.get_config(),
        'initial_guess': csv_importer.get_initial_guess(),
    }

    csv_importer = ImportCSV("firm_ce/settings")
    settings = csv_importer.get_setting('settings.csv')
    data['settings'] = {}
    for idx in range(len(settings)):
        data['settings'][settings[idx]['name']] = csv_importer.get_setting(settings[idx]['filename'])

    return data
