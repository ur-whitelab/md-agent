import json
import os
import time
import warnings
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mdagent.tools.base_tools import (
    CleaningTools,
    SimulationFunctions,
    VisFunctions,
    get_pdb,
)
from mdagent.tools.base_tools.analysis_tools.plot_tools import PlottingTools
from mdagent.tools.base_tools.preprocess_tools.packing import PackMolTool
from mdagent.tools.base_tools.preprocess_tools.pdb_get import MolPDB
from mdagent.utils import FileType, PathRegistry

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
def fibronectin():
    return "fibronectin pdb"


@pytest.fixture
def get_registry():
    return PathRegistry()


@pytest.fixture
def sim_fxns(get_registry):
    return SimulationFunctions(get_registry)


@pytest.fixture
def plotting_tools(get_registry):
    return PlottingTools(get_registry)


@pytest.fixture
def vis_fxns(get_registry):
    return VisFunctions(get_registry)


@pytest.fixture
def packmol(get_registry):
    return PackMolTool(get_registry)


@pytest.fixture
def molpdb(get_registry):
    return MolPDB(get_registry)


@pytest.fixture
def cleaning_fxns(get_registry):
    return CleaningTools(get_registry)


def test_process_csv(plotting_tools):
    mock_csv_content = "Time,Value1,Value2\n1,10,20\n2,15,25"
    mock_reader = MagicMock()
    mock_reader.fieldnames = ["Time", "Value1", "Value2"]
    mock_reader.__iter__.return_value = iter(
        [
            {"Time": "1", "Value1": "10", "Value2": "20"},
            {"Time": "2", "Value1": "15", "Value2": "25"},
        ]
    )
    plotting_tools.file_path = "mock_file.csv"
    plotting_tools.file_name = "mock_file.csv"
    with patch("builtins.open", mock_open(read_data=mock_csv_content)):
        with patch("csv.DictReader", return_value=mock_reader):
            plotting_tools.process_csv()

    assert plotting_tools.headers == ["Time", "Value1", "Value2"]
    assert len(plotting_tools.matched_headers) == 1
    assert plotting_tools.matched_headers[0][1] == "Time"
    assert len(plotting_tools.data) == 2
    assert (
        plotting_tools.data[0]["Time"] == "1"
        and plotting_tools.data[0]["Value1"] == "10"
    )


def test_plot_data(plotting_tools):
    # Test successful plot generation
    data_success = [
        {"Time": "1", "Value1": "10", "Value2": "20"},
        {"Time": "2", "Value1": "15", "Value2": "25"},
    ]
    headers = ["Time", "Value1", "Value2"]
    matched_headers = [(0, "Time")]

    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
        "matplotlib.pyplot.xlabel"
    ), patch("matplotlib.pyplot.ylabel"), patch("matplotlib.pyplot.title"), patch(
        "matplotlib.pyplot.savefig"
    ), patch(
        "matplotlib.pyplot.close"
    ):
        plotting_tools.data = data_success
        plotting_tools.headers = headers
        plotting_tools.matched_headers = matched_headers
        created_plots = plotting_tools.plot_data()
        assert "time_vs_value1.png" in created_plots
        assert "time_vs_value2.png" in created_plots

    # Test failure due to non-numeric data
    data_failure = [
        {"Time": "1", "Value1": "A", "Value2": "B"},
        {"Time": "2", "Value1": "C", "Value2": "D"},
    ]

    plotting_tools.data = data_failure
    plotting_tools.headers = headers
    plotting_tools.matched_headers = matched_headers

    with pytest.raises(Exception) as excinfo:
        plotting_tools.plot_data()
        assert "All plots failed due to non-numeric data." in str(excinfo.value)


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(path_to_cif, vis_fxns):
    result = vis_fxns.run_molrender(path_to_cif)
    assert result == "Visualization created"


def test_find_png(vis_fxns):
    vis_fxns.starting_files = os.listdir(".")
    test_file = "test_image.png"
    with open(test_file, "w") as f:
        f.write("")
    png_files = vis_fxns._find_png()
    assert test_file in png_files

    os.remove(test_file)


def test_create_notebook(path_to_cif, vis_fxns):
    result = vis_fxns.create_notebook(path_to_cif)
    path_to_notebook = path_to_cif.split(".")[0] + "_vis.ipynb"
    os.remove(path_to_notebook)
    assert result == "Visualization Complete"


def test_add_hydrogens_and_remove_water(path_to_cif, cleaning_fxns):
    result = cleaning_fxns._add_hydrogens_and_remove_water(path_to_cif)
    path_to_cleaned_file = "tidy_" + path_to_cif
    os.remove(path_to_cleaned_file)
    assert "Cleaned File" in result


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


def test_getpdb(fibronectin, get_registry):
    name, _ = get_pdb(fibronectin, get_registry)
    assert name.endswith(".pdb")


@pytest.fixture
def path_registry():
    registry = PathRegistry()
    registry.get_timestamp = lambda: "20240109"
    return registry


def test_write_to_file():
    path_registry = PathRegistry()

    with patch("builtins.open", mock_open()):
        file_name = path_registry.write_file_name(
            FileType.PROTEIN,
            protein_name="1XYZ",
            description="testing",
            file_format="pdb",
        )
        # assert file name starts and ends correctly
        assert file_name.startswith("1XYZ")
        assert file_name.endswith(".pdb")


def test_write_file_name_protein(path_registry):
    file_name = path_registry.write_file_name(
        FileType.PROTEIN, protein_name="1XYZ", description="testing", file_format="pdb"
    )
    assert file_name == "1XYZ_testing_20240109.pdb"


def test_write_file_name_simulation_with_conditions(path_registry):
    file_name = path_registry.write_file_name(
        FileType.SIMULATION,
        type_of_sim="MD",
        protein_file_id="1XYZ",
        conditions="pH7",
        time_stamp="20240109",
    )
    assert file_name == "MD_1XYZ_pH7_20240109.py"


def test_write_file_name_simulation_modified(path_registry):
    file_name = path_registry.write_file_name(
        FileType.SIMULATION, Sim_id="SIM456", modified=True, time_stamp="20240109"
    )
    assert file_name == "SIM456_MOD_20240109.py"


def test_write_file_name_simulation_default(path_registry):
    file_name = path_registry.write_file_name(
        FileType.SIMULATION,
        type_of_sim="MD",
        protein_file_id="123",
        time_stamp="20240109",
    )
    assert file_name == "MD_123_20240109.py"


def test_write_file_name_record(path_registry):
    file_name = path_registry.write_file_name(
        FileType.RECORD,
        record_type="REC",
        protein_file_id="123",
        Sim_id="SIM456",
        term="dcd",
        time_stamp="20240109",
    )
    assert file_name == "REC_SIM456_123_20240109.dcd"


def test_map_path():
    mock_json_data = {
        "existing_name": {
            "path": "existing/path",
            "name": "path",
            "description": "Existing description",
        }
    }
    new_path_dict = {
        "new_name": {
            "path": "new/path",
            "name": "path",
            "description": "New description",
        }
    }
    updated_json_data = {**mock_json_data, **new_path_dict}

    path_registry = PathRegistry()
    path_registry.json_file_path = "dummy_json_file.json"

    # Mocking os.path.exists to simulate the JSON file existence
    with patch("os.path.exists", return_value=True):
        # Mocking open for both reading and writing the JSON file
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_json_data))
        ) as mocked_file:
            # Optionally, you can mock internal methods if needed
            with patch.object(
                path_registry, "_check_for_json", return_value=True
            ), patch.object(
                path_registry, "_check_json_content", return_value=True
            ), patch.object(
                path_registry, "_get_full_path", return_value="new/path"
            ):  # Mocking _get_full_path
                result = path_registry.map_path(
                    "new_name", "new/path", "New description"
                )
                # Aggregating all calls to write into a single string
                written_data = "".join(
                    call.args[0] for call in mocked_file().write.call_args_list
                )

                # Comparing the aggregated data with the expected JSON data
                assert json.loads(written_data) == updated_json_data

                # Check the result message
                assert result == "Path successfully mapped to name: new_name"


def test_small_molecule_pdb(molpdb):
    # Test with a valid SMILES string
    valid_smiles = "C1=CC=CC=C1"  # Benzene
    expected_output = (
        "PDB file for C1=CC=CC=C1 successfully created and saved to "
        "files/pdb/benzene.pdb."
    )
    assert molpdb.small_molecule_pdb(valid_smiles) == expected_output
    assert os.path.exists("files/pdb/benzene.pdb")
    os.remove("files/pdb/benzene.pdb")  # Clean up

    # test with invalid SMILES string and invalid molecule name
    invalid_smiles = "C1=CC=CC=C1X"
    invalid_name = "NotAMolecule"
    expected_output = (
        "There was an error getting pdb. Please input a single molecule name."
    )
    assert molpdb.small_molecule_pdb(invalid_smiles) == expected_output
    assert molpdb.small_molecule_pdb(invalid_name) == expected_output

    # test with valid molecule name
    valid_name = "water"
    expected_output = (
        "PDB file for water successfully created and " "saved to files/pdb/water.pdb."
    )
    assert molpdb.small_molecule_pdb(valid_name) == expected_output
    assert os.path.exists("files/pdb/water.pdb")
    os.remove("files/pdb/water.pdb")  # Clean up


def test_packmol_sm_download_called(packmol):
    packmol.path_registry.map_path("1A3N_144150", "files/pdb/1A3N_144150.pdb", "pdb")
    with patch(
        "mdagent.tools.base_tools.preprocess_tools.packing.PackMolTool._get_sm_pdbs",
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


def test_packmol_download_only(packmol):
    packmol.path_registry._remove_path_from_json("water")
    packmol.path_registry._remove_path_from_json("benzene")
    small_molecules = ["water", "benzene"]
    packmol._get_sm_pdbs(small_molecules)
    assert os.path.exists("files/pdb/water.pdb")
    assert os.path.exists("files/pdb/benzene.pdb")
    os.remove("files/pdb/water.pdb")
    os.remove("files/pdb/benzene.pdb")


def test_packmol_download_only_once(packmol):
    packmol.path_registry._remove_path_from_json("water")
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


mocked_files = {"files/solvents": ["water.pdb"]}


def mock_exists(path):
    return path in mocked_files


def mock_listdir(path):
    return mocked_files.get(path, [])


@pytest.fixture
def path_registry_with_mocked_fs():
    with patch("os.path.exists", side_effect=mock_exists):
        with patch("os.listdir", side_effect=mock_listdir):
            registry = PathRegistry()
            registry.get_timestamp = lambda: "20240109"
            return registry


def test_init_path_registry(path_registry_with_mocked_fs):
    # This test will run with the mocked file system
    # Here, you can assert if 'water.pdb' under 'solvents' is registered correctly
    # Depending on how your PathRegistry class stores the registry,
    # you may need to check the internal state or the contents of the JSON file.
    # For example:
    assert "water_000000" in path_registry_with_mocked_fs.list_path_names()
