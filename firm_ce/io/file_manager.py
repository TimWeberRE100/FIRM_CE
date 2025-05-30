from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

class ImportCSV:
    """
    Class for importing CSV files from a given repository.
    """

    def __init__(self, repository: Path) -> None:
        """
        Initialize the importer with a path to the repository.

        Parameters:
        -------
        repository (Path): Path to the directory containing CSV files.
        """

        self.repository = Path(repository)

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def get_data(self, filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Load a CSV file into a nested dictionary using 'id' as the index.

        Parameters:
        -------
        filename (str): Name of the CSV file to load.

        Returns:
        -------
        Dict[str, Dict[str, Any]]: Dictionary of records keyed by ID.
        """

        filepath = self.repository.joinpath(filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        return pd.read_csv(filepath, index_col="id").to_dict(orient='index')
    
    def get_scenarios(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("scenarios.csv")
    
    def get_generators(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("generators.csv")  
    
    def get_fuels(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("fuels.csv")  
    
    def get_lines(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("lines.csv")
    
    def get_storages(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("storages.csv")
    
    def get_datafiles(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("datafiles.csv")
    
    def get_config(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("config.csv")
    
    def get_initial_guess(self) -> Dict[str, Dict[str, Any]]:
        return self.get_data("initial_guess.csv")

    def get_setting(self, setting_filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Load a specified settings CSV file. Settings should NOT be edited by user.

        Parameters:
        -------
        setting_filename (str): Name of the settings CSV file.

        Returns:
        -------
        Dict[str, Dict[str, Any]]: Dictionary of settings keyed by ID.
        """

        return self.get_data(setting_filename)
        
class ImportDatafile:
    """
    Class for importing a single CSV datafile into a dictionary of lists.
    """

    def __init__(self, repository: Path, filename: str) -> None:
        """
        Initialize the datafile importer.

        Parameters:
        -------
        repository (Path): Directory containing the CSV file.
        filename (str): Name of the CSV datafile to import.
        """

        self.repository = Path(repository)
        self.filename = filename

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def get_data(self) -> Dict[str, List[Any]]:
        """
        Load the CSV data into a dictionary of column-wise lists.

        Returns:
        -------
        Dict[str, List[Any]]: Dictionary where keys are column names and values are column data.
        """

        filepath = self.repository.joinpath(self.filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        df = pd.read_csv(filepath)
        return df.to_dict(orient='list')

class DataFile:
    """
    Container for a named datafile and its content.
    """

    def __init__(self, filename: str, datafile_type: str) -> None:
        """
        Initialize the DataFile with a filename and its type.

        Parameters:
        -------
        filename (str): Name of the datafile.
        datafile_type (str): Descriptive type of the datafile (e.g., 'time series').
        """
        self.name = filename
        self.type = datafile_type
        self.data = ImportDatafile("firm_ce/data", filename).get_data()

    def __repr__(self) -> str:
        return f"{self.name} ({self.type})"

def import_csv_data() -> Dict[str, Any]:
    """
    Load all model configuration CSVs into a single dictionary.

    Returns:
    -------
    Dict[str, Any]: A dictionary containing model configuration data.
    """

    csv_importer = ImportCSV("firm_ce/config")
    data = {
        'scenarios': csv_importer.get_scenarios(),
        'generators': csv_importer.get_generators(),
        'fuels': csv_importer.get_fuels(),
        'lines': csv_importer.get_lines(),
        'storages': csv_importer.get_storages(),
        'config': csv_importer.get_config(),
        'initial_guess': csv_importer.get_initial_guess(),
    }

    csv_importer = ImportCSV("firm_ce/settings")
    settings = csv_importer.get_setting('settings.csv')
    data['settings'] = {}
    for idx in range(len(settings)):
        data['settings'][settings[idx]['name']] = csv_importer.get_setting(settings[idx]['filename'])

    return data

def import_datafiles() -> Dict[str, Dict[str, Any]]:
    """
    Load the datafiles.csv model configuration into a dictionary.

    Returns:
    -------
    Dict[str, Dict[str, Any]]: Dictionary of datafiles keyed by their ID.
    """

    csv_importer = ImportCSV("firm_ce/config")
    data = csv_importer.get_datafiles()

    return data
