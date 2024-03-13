import json
import os
import tempfile
from datetime import datetime
from unittest.mock import mock_open, patch

import pytest
from langchain.chat_models import ChatOpenAI

from mdagent.tools.base_tools import Scholar2ResultLLM
from mdagent.utils import FileType, PathRegistry


@pytest.fixture
def todays_date():
    return str(datetime.today().strftime("%Y%m%d"))


@pytest.fixture
def path_registry():
    return PathRegistry()


def test_write_to_file(path_registry):
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


def test_write_file_name_protein(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.PROTEIN, protein_name="1XYZ", description="testing", file_format="pdb"
    )
    assert "1XYZ_testing" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".pdb")


def test_write_file_name_simulation_with_conditions(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.SIMULATION,
        type_of_sim="MD",
        protein_file_id="1XYZ",
        conditions="pH7",
        time_stamp=todays_date,
    )
    assert "MD_1XYZ_pH7" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".py")


def test_write_file_name_simulation_modified(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.SIMULATION, Sim_id="SIM456", modified=True, time_stamp=todays_date
    )
    assert "SIM456_MOD" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".py")


def test_write_file_name_simulation_default(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.SIMULATION,
        type_of_sim="MD",
        protein_file_id="123",
        time_stamp=todays_date,
    )
    assert "MD_123" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".py")


def test_write_file_name_record(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.RECORD,
        record_type="REC",
        protein_file_id="123",
        Sim_id="SIM456",
        term="dcd",
        time_stamp=todays_date,
    )
    assert "REC_SIM456_123" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".dcd")


def test_write_file_name_figure_1(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.FIGURE,
        protein_file_id="123",
        Sim_id="SIM456",
        time_stamp=todays_date,
        irrelevant="irrelevant",
    )
    assert "FIG_SIM456_123_" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".png")


def test_write_file_name_figure_2(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.FIGURE,
        protein_file_id="X123",
        Log_id="LOG_123456",
        time_stamp=todays_date,
        file_format="jpg",
        irrelevant="irrelevant",
    )
    assert "FIG_LOG_123456_X123_" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".jpg")


def test_write_file_name_figure_3(path_registry, todays_date):
    file_name = path_registry.write_file_name(
        FileType.FIGURE,
        protein_file_id="X123",
        Log_id="LOG_123456",
        fig_analysis="randomanalytic",
        file_format="jpg",
        irrelevant="irrelevant",
    )
    assert "FIG_randomanalytic_LOG_123456_" in file_name
    assert todays_date in file_name
    assert file_name.endswith(".jpg")


def test_map_path(path_registry):
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


def test_init_path_registry(path_registry):
    temp_file, temp_path = tempfile.mkstemp()
    path_registry.map_path("temp_path", str(temp_path), "temp file")
    assert "temp_path" in path_registry.list_path_names()
    os.close(temp_file)
    os.remove(temp_path)


@pytest.fixture
def questions():
    qs = [
        "What are the effects of norhalichondrin B in mammals?",
    ]
    return qs[0]


@pytest.mark.skip(reason="This requires an API call")
def test_litsearch(questions):
    llm = ChatOpenAI()

    searchtool = Scholar2ResultLLM(llm=llm)
    for q in questions:
        ans = searchtool._run(q)
        assert isinstance(ans, str)
        assert len(ans) > 0
    if os.path.exists("../query"):
        os.rmdir("../query")
