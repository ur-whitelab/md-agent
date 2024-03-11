from unittest.mock import mock_open, patch

import pytest

from mdagent.tools.base_tools import SimulationFunctions
from mdagent.utils import PathRegistry


@pytest.fixture
def get_registry():
    return PathRegistry()


@pytest.fixture
def sim_fxns(get_registry):
    return SimulationFunctions(get_registry)


@patch("os.path.exists")
@patch("os.listdir")
def test_extract_parameters_path(mock_listdir, mock_exists, sim_fxns):
    # Test when parameters.json exists
    mock_exists.return_value = True
    assert sim_fxns._extract_parameters_path() == "simulation_parameters_summary.json"
    mock_exists.assert_called_once_with("simulation_parameters_summary.json")
    mock_exists.reset_mock()  # Reset the mock for the next scenario

    # Test when parameters.json does not exist, but some_parameters.json does
    mock_exists.return_value = False
    mock_listdir.return_value = ["some_parameters.json", "other_file.txt"]
    assert sim_fxns._extract_parameters_path() == "some_parameters.json"

    # Test when no appropriate file exists
    mock_listdir.return_value = ["other_file.json", "other_file.txt"]
    with pytest.raises(ValueError) as e:
        sim_fxns._extract_parameters_path()
    assert str(e.value) == "No parameters.json file found in directory."


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"param1": "value1", "param2": "value2"}',
)
@patch("json.load")
def test_setup_simulation_from_json(mock_json_load, mock_file_open, sim_fxns):
    # Define the mock behavior for json.load
    mock_json_load.return_value = {"param1": "value1", "param2": "value2"}
    params = sim_fxns._setup_simulation_from_json("test_file.json")
    mock_file_open.assert_called_once_with("test_file.json", "r")
    mock_json_load.assert_called_once()
    assert params == {"param1": "value1", "param2": "value2"}
