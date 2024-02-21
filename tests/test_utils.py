import json
import warnings
from unittest.mock import mock_open, patch

import pytest

from mdagent.utils import FileType, PathRegistry

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


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
