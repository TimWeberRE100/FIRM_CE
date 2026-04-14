import csv
import shutil
from pathlib import Path

import pytest

from firm_ce.model import Model


def _update_config_type(config_csv: Path, config_type: str) -> None:
    with config_csv.open(newline="") as infile:
        rows = list(csv.DictReader(infile))

    for row in rows:
        if row["name"] == "type":
            row["value"] = config_type

    with config_csv.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["id", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _update_scenario_type(scenarios_csv: Path, config_type: str) -> None:
    with scenarios_csv.open(newline="") as infile:
        rows = list(csv.DictReader(infile))
        fieldnames = list(rows[0].keys())

    for row in rows:
        row["type"] = config_type

    with scenarios_csv.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.slow
def test_capacity_expansion_integration(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests" / "inputs" / "test_1hr_config_data"
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"

    shutil.copytree(fixture_root / "config", config_dir)
    shutil.copytree(fixture_root / "data", data_dir)

    _update_config_type(config_dir / "config.csv", "capacity_expansion")
    _update_scenario_type(config_dir / "scenarios.csv", "capacity_expansion")

    model = Model(
        config_directory=str(config_dir),
        data_directory=str(data_dir),
        logging_flag=False,
    )
    model.solve()

    scenario = model.scenarios["gas"]
    pathway_dir = Path(scenario.results_dir) / f"{scenario.name}_capacity_expansion" / "pathway"

    expected_files = [
        "capacity_expansion_metrics.csv",
        "generators_new_build.csv",
        "generators_cumulative_capacity.csv",
        "storages_power_new_build.csv",
        "storages_power_cumulative_capacity.csv",
        "storages_energy_new_build.csv",
        "storages_energy_cumulative_capacity.csv",
        "lines_new_build.csv",
        "lines_cumulative_capacity.csv",
    ]

    for filename in expected_files:
        assert (pathway_dir / filename).exists()

    with (pathway_dir / "capacity_expansion_metrics.csv").open(newline="") as infile:
        rows = list(csv.DictReader(infile))

    assert len(rows) == 1
    assert rows[0]["year"] == "2020"
