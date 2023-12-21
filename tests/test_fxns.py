import os
import warnings
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mdagent.tools.base_tools import (
    CleaningTools,
    SimulationFunctions,
    VisFunctions,
    get_pdb,
)
from mdagent.tools.base_tools.analysis_tools.plot_tools import process_csv
from mdagent.utils import PathRegistry

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


@pytest.fixture
def path_to_cif():
    # Save original working directory
    original_cwd = os.getcwd()

    # Change current working directory to the directory where the CIF file is located
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tests_dir)

    # Yield the filename only
    filename_only = "3pqr.cif"
    yield filename_only

    # Restore original working directory after the test is done
    os.chdir(original_cwd)


@pytest.fixture
def cleaning_fxns():
    return CleaningTools()


# Test simulation tools
@pytest.fixture
def sim_fxns():
    return SimulationFunctions()


# Test visualization tools
@pytest.fixture
def vis_fxns():
    return VisFunctions()


# Test MD utility tools
@pytest.fixture
def fibronectin():
    return "fibronectin pdb"


@pytest.fixture
def get_registry():
    return PathRegistry()


def test_process_csv():
    mock_csv_content = "Time,Value1,Value2\n1,10,20\n2,15,25"
    mock_reader = MagicMock()
    mock_reader.fieldnames = ["Time", "Value1", "Value2"]
    mock_reader.__iter__.return_value = iter(
        [
            {"Time": "1", "Value1": "10", "Value2": "20"},
            {"Time": "2", "Value1": "15", "Value2": "25"},
        ]
    )

    with patch("builtins.open", mock_open(read_data=mock_csv_content)):
        with patch("csv.DictReader", return_value=mock_reader):
            data, headers, matched_headers = process_csv("mock_file.csv")

    # Assertions
    assert headers == ["Time", "Value1", "Value2"]
    assert len(matched_headers) == 1
    assert matched_headers[0][1] == "Time"
    assert len(data) == 2
    assert data[0]["Time"] == "1" and data[0]["Value1"] == "10"


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(path_to_cif, vis_fxns):
    result = vis_fxns.run_molrender(path_to_cif)
    assert result == "Visualization created"


def test_create_notebook(path_to_cif, vis_fxns, get_registry):
    result = vis_fxns.create_notebook(path_to_cif, get_registry)
    assert result == "Visualization Complete"


def test_add_hydrogens_and_remove_water(path_to_cif, cleaning_fxns, get_registry):
    result = cleaning_fxns._add_hydrogens_and_remove_water(path_to_cif, get_registry)
    assert "Cleaned File" in result  # just want to make sur the function ran


@patch("os.path.exists")
@patch("os.listdir")
def test_extract_parameters_path(mock_listdir, mock_exists, sim_fxns, get_registry):
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


def test_getpdb(fibronectin, get_registry):
    name = get_pdb(fibronectin, get_registry)
    assert name.endswith(".pdb")
