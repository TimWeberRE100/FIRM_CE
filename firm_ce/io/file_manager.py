from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import numpy as np
from numpy.typing import NDArray

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
        self.config_filenames = ('scenarios',
                                'generators',
                                'fuels',
                                'lines',
                                'storages',
                                'config',
                                'initial_guess',
                                'datafiles',)

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
        imported_dict = pd.read_csv(filepath, index_col="id").to_dict(orient='index')
        for idx in imported_dict:
            imported_dict[idx]['id'] = idx
        return imported_dict
    
    def get_config_dict(self) -> Dict[str, Dict[str, Any]]:
        return {fn : self.get_data(fn+'.csv') for fn in self.config_filenames}
        
class ImportDatafile:
    """
    Class for importing a single CSV datafile into a dictionary of NDArrays.
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
        
    def __repr__(self) -> str:
        return f"ImportDatafile ({self.filename!r})"

    def get_data(self) -> Dict[str, NDArray]:
        """
        Load the CSV data into a dictionary of column-wise NumPy arrays.

        Returns:
        -------
        Dict[str, NDArray]: Dictionary where keys are column names and values are NumPy arrays.
        """
        filepath = self.repository.joinpath(self.filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        df = pd.read_csv(filepath)
        return {col: df[col].to_numpy() for col in df.columns}

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
        return f"DataFile ({self.name!r}, {self.type!r}, {self.data!r})"

def import_config_csvs() -> Dict[str, Any]:
    """
    Load all model configuration CSVs into a single dictionary.

    Returns:
    -------
    Dict[str, Any]: A dictionary containing model configuration data.
    """

    csv_importer = ImportCSV("firm_ce/config")
    data = csv_importer.get_config_dict()

    return data
