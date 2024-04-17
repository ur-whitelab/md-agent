import json
import os

import pytest

from mdagent.subagents.agents import MemoryManager
from mdagent.subagents.subagent_setup import SubAgentInitializer, SubAgentSettings


@pytest.fixture
def memory_manager(get_registry):
    return MemoryManager(get_registry("raw", False))


def test_subagent_setup(get_registry):
    settings = SubAgentSettings(get_registry("raw", False))
    initializer = SubAgentInitializer(settings)
    subagents = initializer.create_iteration_agents()
    action = subagents["action"]
    skill = subagents["skill"]
    critic = subagents["critic"]
    curriculum = initializer.create_curriculum()
    assert action is not None
    assert critic is not None
    assert curriculum is not None
    assert skill is not None


def test_memory_init(memory_manager, get_registry):
    assert memory_manager is not None
    assert memory_manager.run_id is not None
    assert len(memory_manager.run_id) == 8
    assert os.path.exists(memory_manager.cnt_history_dir)
    assert os.path.exists(memory_manager.cnt_history_details_dir)

    mm_path_id = MemoryManager(get_registry("raw", False), run_id="TESTRUNN")
    assert mm_path_id.run_id == "TESTRUNN"


def test_write_to_and_retrieve_from_history_cnt(memory_manager):
    input_dict = {
        "prompt": "prompt_",
        "code": "code_",
        "output": "output_",
        "critique": "critique_",
        "success": True,
    }
    memory_manager._write_history_iterator(**input_dict)
    assert os.path.exists(memory_manager.cnt_history_details)
    with open(memory_manager.cnt_history_details, "r") as f:
        data = json.load(f)
    input_dict["summary"] = None
    assert data["0.0"] == input_dict

    memory = memory_manager.retrieve_recent_memory_iterator(last_only=True)
    assert str(memory) == str(input_dict)


def test_write_to_json_new_file(tmp_path, memory_manager):
    file_path = tmp_path / "test.json"
    test_data = {"key": "value"}
    memory_manager._write_to_json(test_data, str(file_path))
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == test_data


def test_write_to_json_existing_file(tmp_path, memory_manager):
    file_path = tmp_path / "test.json"
    initial_data = {"initial_key": "initial_value"}
    update_data = {"updated_key": "updated_value"}
    with open(file_path, "w") as f:
        json.dump(initial_data, f)

    memory_manager._write_to_json(update_data, str(file_path))
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == {**initial_data, **update_data}


def test_pull_memory_summary(get_registry):
    mm_mem = MemoryManager(get_registry("raw", False), run_id="TESTRUNN")
    fake_summaries = {"TESTRUNN.0": "fake_summary"}
    with open(mm_mem.agent_trace_summary, "w") as f:
        f.write(json.dumps(fake_summaries))
    output = mm_mem.pull_agent_summary_from_mem(run_id="TESTRUNN")
    assert output == "fake_summary"
    assert mm_mem.pull_agent_summary_from_mem(run_id="TESTRUNN") == "fake_summary"
