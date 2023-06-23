import os
import warnings
from unittest.mock import mock_open, patch

import pytest

from mdagent.tools.clean_tools import _add_hydrogens_and_remove_water
from mdagent.tools.md_util_tools import get_pdb
from mdagent.tools.setup_and_run import SimulationFunctions
from mdagent.tools.vis_tools import VisFunctions

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
    return "fibronectin"


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(
    path_to_cif,
):
    result = vis_fxns.run_molrender(path_to_cif, vis_fxns)
    assert result == "Visualization created"


def test_create_notebook(path_to_cif, vis_fxns):
    result = vis_fxns.create_notebook(path_to_cif)
    assert result == "Visualization Complete"


def test_add_hydrogens_and_remove_water(path_to_cif):
    result = _add_hydrogens_and_remove_water(path_to_cif)
    assert "Cleaned File" in result  # just want to make sur the function ran


@patch("os.path.exists")
@patch("os.listdir")
def test_extract_parameters_path(sim_fxns, mock_listdir, mock_exists):
    # Test when parameters.json exists
    mock_exists.return_value = True
    assert sim_fxns._extract_parameters_path() == "parameters.json"
    mock_exists.assert_called_once_with("parameters.json")
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
def test_setup_simulation_from_json(mock_json_load, mock_file_open):
    # Define the mock behavior for json.load
    mock_json_load.return_value = {"param1": "value1", "param2": "value2"}
    params = sim_fxns._setup_simulation_from_json("test_file.json")
    mock_file_open.assert_called_once_with("test_file.json", "r")
    mock_json_load.assert_called_once()
    assert params == {"param1": "value1", "param2": "value2"}


def test_getpdb(fibronectin):
    name = get_pdb(fibronectin)
    assert name == "1X3D.cif"
