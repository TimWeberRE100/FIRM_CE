from pathlib import Path
import pandas as pd

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
    
    def get_lines(self) -> dict:
        return self.get_data("lines.csv")
    
    def get_storages(self) -> dict:
        return self.get_data("storages.csv")
    
    def get_constraints(self) -> dict:
        return self.get_data("constraints.csv")
    
class DataSheet:
    pass
    
def import_csv_data() -> dict:
    csv_importer = ImportCSV("firm_ce/config")
    data = {
        'scenarios': csv_importer.get_scenarios(),
        'generators': csv_importer.get_generators(),
        'lines': csv_importer.get_lines(),
        'storages': csv_importer.get_storages(),
        'constraints': csv_importer.get_constraints(),
    }

    return data