import pytest
from firm_ce.model import Model


@pytest.mark.slow
def test_end_to_end_integration():
    model = Model(config_directory="tests/test_24hr_config_data/config")
    model.solve()
