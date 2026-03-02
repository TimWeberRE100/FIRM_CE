import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from firm_ce.common.constants import YEAR_ALL_STR, MULTIYEAR_FILENAMES


class ImportCSV:
    """
    Class for importing CSV files from a given repository.

    This class is designed to handle model configuration files such as
    scenarios, generators, fuels, lines, storages, and others. Each CSV
    is loaded into a nested dictionary indexed by the `id` column.

    Files in MULTIYEAR_FILENAMES are split across two CSVs:
    - `{name}.csv` contains static asset metadata with one row per asset.
    - `{name}_multiyear.csv` contains year-varying fields (e.g., costs, capacities etc.)
    with one row per (asset, year).
    The two files are merged by `get_merged_multiyear_data` before being
    returned. The resulting structure is `Dict[int, Dict[int|str, Dict[str, Any]]]`
    {asset id : {year : {column : value}}}.
    """

    def __init__(self, repository: Path) -> None:
        """
        Initialize the importer with a path to the repository.

        Parameters:
        -------
        repository (Path): Path to the directory containing CSV files.

        Raises
        ------
        FileNotFoundError: If the specified repository path does not exist.
        """

        self.repository = Path(repository)
        self.config_filenames = (
            "scenarios",
            "nodes",
            "generators",
            "fuels",
            "lines",
            "storages",
            "config",
            "initial_guess",
            "datafiles",
        )

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def get_static_data(self, filename: str) -> Dict[int, Dict[str, Any]]:
        """
        Load a CSV file into a nested dictionary keyed by 'id'.

        Parameters:
        -------
        filename (str): Name of the CSV file to load, including the extension.

        Returns:
        -------
        Dict[int, Dict[str, Any]]: Dictionary of records keyed by integer 'id'.

        Raises:
        -------
        FileNotFoundError: If the specified repository path does not exist.
        """

        filepath = self.repository.joinpath(filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        raw_dict = pd.read_csv(filepath, index_col="id").to_dict(orient="index")
        imported_dict = {int(k): v for k, v in raw_dict.items()}
        for idx in imported_dict:
            imported_dict[idx]["id"] = idx
        return imported_dict

    def get_multiyear_data(self, filename: str) -> Dict[int, Dict[int | str, Dict[str, Any]]]:
        """
        Load a CSV file that contains a 'year' column into a nested dictionary grouped
        first by asset id and then by year.

        Each unique (asset id, year) pair becomes one entry. The returned structure is:
            {asset id : {year : {column : value}}}
        The 'id' values are stored as integers. Year values are stored as integers for
        specific years, or as the string YEAR_ALL_STR when the CSV cell contains YEAR_ALL_STR
        (case-insensitive). Rows with year==YEAR_ALL_STR apply to every year in the modelling
        horizon and are expanded to actual year keys by expand_year_dict before use.

        Parameters:
        -------
        filename (str): Name of the CSV file to load.

        Returns:
        -------
        Dict[int, Dict[int | str, Dict[str, Any]]]: Nested dictionary keyed by asset
            id then year (integer) or YEAR_ALL_STR (string).

        Exceptions:
        -------
        FileNotFoundError: If the specified file does not exist.
        """
        filepath = self.repository.joinpath(filename)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        df = pd.read_csv(filepath)
        multiyear_dict = {}
        for _, row in df.iterrows():
            asset_id = int(row["id"])
            year_raw = str(row["year"]).strip()
            year = YEAR_ALL_STR if year_raw.lower() == YEAR_ALL_STR else int(row["year"])
            row_dict = row.to_dict()
            row_dict["id"] = asset_id
            if asset_id not in multiyear_dict:
                multiyear_dict[asset_id] = {}
            multiyear_dict[asset_id][year] = row_dict
        return multiyear_dict

    def get_merged_multiyear_data(self, base_name: str) -> Dict[int, Dict[int | str, Dict[str, Any]]]:
        """
        Load and merge the static and multiyear CSVs for a MULTIYEAR_FILENAMES entry.

        Loads `{base_name}.csv` (static fields, one row per asset) and
        `{base_name}_multiyear.csv` (year-varying fields, one row per asset + year),
        then add static fields into every year's dict for each asset.

        Parameters:
        -------
        base_name (str): Filename without extension (e.g. "generators").

        Returns:
        -------
        Dict[int, Dict[int | str, Dict[str, Any]]]: Merged dict keyed by asset id then
            year (integer) or YEAR_ALL_STR (string), with both static and year-varying fields
            present in each year's inner dict.

        Exceptions:
        -------
        ValueError: If an asset id appears in the static file but not in the multiyear
            file, or vice versa.
        FileNotFoundError: If either CSV file does not exist.
        """
        static_data = self.get_static_data(base_name + ".csv")
        multiyear_data = self.get_multiyear_data(base_name + "_multiyear.csv")

        static_ids = set(static_data.keys())
        multiyear_ids = set(multiyear_data.keys())

        only_static = static_ids - multiyear_ids
        only_multiyear = multiyear_ids - static_ids
        if only_static:
            raise ValueError(
                f"{base_name}_multiyear.csv is missing entries for asset id(s) {sorted(only_static)} "
                f"that are defined in {base_name}.csv."
            )
        if only_multiyear:
            raise ValueError(
                f"{base_name}.csv is missing entries for asset id(s) {sorted(only_multiyear)} "
                f"that are defined in {base_name}_multiyear.csv."
            )

        for asset_id, year_dict in multiyear_data.items():
            static_row = static_data[asset_id]
            for year_data in year_dict.values():
                for field, value in static_row.items():
                    if field not in year_data:
                        year_data[field] = value

        return multiyear_data

    def get_config_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all predefined configuration CSVs into a dictionary.

        Files listed in MULTIYEAR_FILENAMES are loaded by merging a static CSV
        (`{name}.csv`) with a multiyear CSV (`{name}_multiyear.csv`) via
        `get_merged_multiyear_data`, returning `Dict[int, Dict[int|str, Dict[str, Any]]]`.
        All other files are loaded via `get_static_data` returning
        `Dict[int, Dict[str, Any]]`.

        Returns
        -------
        Dict[str, Dict[str, Any]]: A dictionary mapping each configuration filename
        (without extension) to its corresponding record dictionary.
        """
        result = {}
        for filename in self.config_filenames:
            if filename in MULTIYEAR_FILENAMES:
                result[filename] = self.get_merged_multiyear_data(filename)
            else:
                result[filename] = self.get_static_data(filename + ".csv")
        return result


def expand_year_dict(
    year_dict: Dict[int | str, Dict[str, Any]], firstyear: int, finalyear: int
) -> Dict[int, Dict[str, Any]]:
    """
    Expand an asset's year-keyed dict so that every year in [firstyear, finalyear]
    has an entry.

    If the dict contains a YEAR_ALL_STR key, its data is copied to every year in the horizon
    that does not already have a specific entry. The YEAR_ALL_STR key is removed from the result.
    Years that already have explicit entries are left unchanged.

    Parameters:
    -------
    year_dict (Dict[int | str, Dict[str, Any]]): Per-asset year dict as returned by
        get_multiyear_data. Keys are year integers or the string YEAR_ALL_STR.
    firstyear (int): First year of the modelling horizon (inclusive).
    finalyear (int): Final year of the modelling horizon (inclusive).

    Returns:
    -------
    Dict[int, Dict[str, Any]]: Expanded year dict with integer keys only, covering every
        year from firstyear to finalyear.
    """
    all_data = year_dict.get(YEAR_ALL_STR)
    result = {k: v for k, v in year_dict.items() if k != YEAR_ALL_STR}

    if all_data is not None:
        for year in range(firstyear, finalyear + 1):
            if year not in result:
                result[year] = {**all_data, "year": year}

    return result


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

        Raises:
        -------
        FileNotFoundError: If the specified repository path does not exist.
        """

        self.repository = Path(repository)
        self.filename = filename

        if not self.repository.is_dir():
            raise FileNotFoundError(f"Repository {repository} does not exist.")

    def __repr__(self) -> str:
        return f"ImportDatafile ({self.filename!r})"

    def get_data(self) -> Dict[str, NDArray]:
        """
        Load the CSV data into a dictionary of NumPy arrays.

        Returns:
        -------
        Dict[str, NDArray]: Dictionary where keys are column names and values are NumPy arrays.

        Raises:
        -------
        FileNotFoundError: If the specified repository path does not exist.
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

    def __init__(self, filename: str, datafile_type: str, file_directory: str) -> None:
        """
        Initialize the DataFile with a filename and its type.

        Parameters:
        -------
        filename (str): Name of the datafile.
        datafile_type (str): Descriptive type of the datafile (e.g., 'time series').
        file_directory (str): Path to the directory containing the datafile.
        """
        self.name = filename
        self.type = datafile_type
        self.file_directory = file_directory
        self.data = ImportDatafile(file_directory, filename).get_data()

    def __repr__(self) -> str:
        return f"DataFile ({self.name!r}, {self.type!r}, {self.data!r})"


class ResultFile:
    """
    Container class for saving model results into a CSV file.

    Handles writing headers, formatting data, and rounding values
    before output.
    """

    def __init__(
        self,
        file_type: str,
        target_directory: str,
        header: List[str],
        data_array: NDArray[np.float64] | NDArray[np.int64],
        decimals: int | None = None,
    ):
        """
        Initialise a result file object.

        Parameters:
        ----------
        file_type (str): Identifier for the file (used as the filename base).
        target_directory (str): Directory where the file will be saved.
        header (List[str]): List of column headers for the CSV file.
        data_array (Union[NDArray[np.float64], NDArray[np.int64]]): Data to
            write into the CSV file.
        decimals (Union[int, None], optional): Number of decimal places to round
            data values to. If None, values are written without rounding.
        """
        self.name = file_type + ".csv"
        self.type = file_type
        self.target_directory = target_directory
        self.header = header
        self.data = data_array
        self.decimals = decimals

    def __repr__(self) -> str:
        return f"ResultFile ({self.type!r})"

    def write(self):
        """
        Write the data to a CSV file.

        Multi-line headers are possible.

        Each row of data_array is written to the file. Optionally,
        values are rounded to the specified number of decimals.

        Returns:
        -------
        None.

        Side-effects:
        ------------
        Creates a CSV file in target_directory with the name
        <file_type>.csv and prints a confirmation message.
        """
        with open(os.path.join(self.target_directory, self.name), mode="w", newline="") as file:
            writer = csv.writer(file)
            if self.header is not None:
                for row in self.header:
                    writer.writerow(row)
            for row in self.data:
                writer.writerow(np.round(row, decimals=self.decimals) if self.decimals is not None else row)
        print(f"Saved {self.name} to {self.target_directory}")
        return None


def import_config_csvs(config_directory: str) -> Dict[str, Any]:
    """
    Load all model configuration CSVs into a single dictionary.

    Parameters:
    ----------
    config_directory (str): Path to the directory containing configuration CSV files.

    Returns:
    -------
    Dict[str, Any]: A dictionary mapping configuration filenames (without extension)
        to their respective record dictionaries.
    """

    csv_importer = ImportCSV(config_directory)
    data = csv_importer.get_config_dict()

    return data


def extract_model_name(config_directory: str) -> str:
    """
    Read config.csv from config_directory and return the model name, defaulting to 'Model'.

    Parameters:
    -------
    config_directory (str): Path to the directory containing `config.csv`.

    Returns:
    -------
    str: The value of the model_name entry in `config.csv`, or "Model" if not present or empty.
    """
    config_dict = import_config_csvs(config_directory).get("config", {})
    for record in config_dict.values():
        if record.get("name") == "model_name":
            model_name = record.get("value")
            if model_name:
                return str(model_name)
    return "Model"
