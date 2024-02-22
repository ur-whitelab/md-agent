import os
import time
import warnings
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mdagent.tools.base_tools import CleaningTools, SimulationFunctions
from mdagent.tools.base_tools.preprocess_tools.pdb_tools import MolPDB, PackMolTool
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


@pytest.fixture
def molpdb():
    return MolPDB()


# Test simulation tools
@pytest.fixture
def sim_fxns():
    return SimulationFunctions()


@pytest.fixture
def get_registry():
    return PathRegistry()


@pytest.fixture
def packmol(get_registry):
    return PackMolTool(get_registry)


def test_add_hydrogens_and_remove_water(path_to_cif, cleaning_fxns, get_registry):
    result = cleaning_fxns._add_hydrogens_and_remove_water(path_to_cif, get_registry)
    assert "Cleaned File" in result  # just want to make sur the function ran


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


def test_small_molecule_pdb(molpdb, get_registry):
    # Test with a valid SMILES string
    valid_smiles = "C1=CC=CC=C1"  # Benzene
    expected_output = (
        "PDB file for C1=CC=CC=C1 successfully created and saved to "
        "files/pdb/benzene.pdb."
    )
    assert molpdb.small_molecule_pdb(valid_smiles, get_registry) == expected_output
    assert os.path.exists("files/pdb/benzene.pdb")
    os.remove("files/pdb/benzene.pdb")  # Clean up

    # test with invalid SMILES string and invalid molecule name
    invalid_smiles = "C1=CC=CC=C1X"
    invalid_name = "NotAMolecule"
    expected_output = (
        "There was an error getting pdb. Please input a single molecule name."
    )
    assert molpdb.small_molecule_pdb(invalid_smiles, get_registry) == expected_output
    assert molpdb.small_molecule_pdb(invalid_name, get_registry) == expected_output

    # test with valid molecule name
    valid_name = "water"
    assert "successfully" in molpdb.small_molecule_pdb(valid_name, get_registry)
    # assert os.path.exists("files/pdb/water.pdb")
    if os.path.exists("files/pdb/water.pdb"):
        os.remove("files/pdb/water.pdb")


def test_packmol_sm_download_called(packmol):
    path_registry = PathRegistry()
    path_registry._remove_path_from_json("water")
    path_registry._remove_path_from_json("benzene")
    path_registry.map_path("1A3N_144150", "files/pdb/1A3N_144150.pdb", "pdb")
    with patch(
        "mdagent.tools.base_tools.preprocess_tools.pdb_tools.PackMolTool._get_sm_pdbs",
        new=MagicMock(),
    ) as mock_get_sm_pdbs:
        test_values = {
            "pdbfiles_id": ["1A3N_144150"],
            "small_molecules": ["water", "benzene"],
            "number_of_molecules": [1, 10, 10],
            "instructions": [
                ["inside box 0. 0. 0. 100. 100. 100."],
                ["inside box 0. 0. 0. 100. 100. 100."],
                ["inside box 0. 0. 0. 100. 100. 100."],
            ],
        }

        packmol._run(**test_values)

        mock_get_sm_pdbs.assert_called_with(["water", "benzene"])


@pytest.mark.skip(reason="Resume this test when ckpt is implemented")
def test_packmol_download_only(packmol):
    path_registry = PathRegistry()
    path_registry._remove_path_from_json("water")
    path_registry._remove_path_from_json("benzene")
    small_molecules = ["water", "benzene"]
    packmol._get_sm_pdbs(small_molecules)
    # assert os.path.exists("files/pdb/water.pdb")
    # assert os.path.exists("files/pdb/benzene.pdb")
    if os.path.exists("files/pdb/water.pdb"):
        os.remove("files/pdb/water.pdb")
    if os.path.exists("files/pdb/benzene.pdb"):
        os.remove("files/pdb/benzene.pdb")


@pytest.mark.skip(reason="Resume this test when ckpt is implemented")
def test_packmol_download_only_once(packmol):
    path_registry = PathRegistry()
    path_registry._remove_path_from_json("water")
    small_molecules = ["water"]
    packmol._get_sm_pdbs(small_molecules)
    assert os.path.exists("files/pdb/water.pdb")
    water_time = os.path.getmtime("files/pdb/water.pdb")
    time.sleep(5)

    # Call the function again with the same molecule
    packmol._get_sm_pdbs(small_molecules)
    water_time_after = os.path.getmtime("files/pdb/water.pdb")

    assert water_time == water_time_after
    # Clean up
    os.remove("files/pdb/water.pdb")
