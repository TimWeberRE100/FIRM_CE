from firm_ce.io.file_manager import import_config_csvs


class ModelData:
    """
    Container for all model configuration data loaded from CSV files.
    """

    def __init__(self, config_directory: str, results_dir: str) -> None:
        """
        Load configuration CSVs into the ModelData container.

        Parameters:
        -------
        config_directory (str): Path to the directory containing all config CSV files.
        results_dir (str): Path to the results directory created for this model run, as returned
            by validate_config.

        Returns:
        -------
        None.
        """
        self.config_directory = config_directory
        self.results_dir = results_dir

        config_data = import_config_csvs(config_directory=config_directory)

        self.scenarios = config_data.get("scenarios")
        self.nodes = config_data.get("nodes")
        self.generators = config_data.get("generators")
        self.fuels = config_data.get("fuels")
        self.lines = config_data.get("lines")
        self.storages = config_data.get("storages")
        self.config = config_data.get("config")
        self.x0s = config_data.get("initial_guess")
        self.datafiles = config_data.get("datafiles")
