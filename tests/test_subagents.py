import json
import os

import pytest
from dotenv import load_dotenv

from mdagent.subagents.agents import MemoryManager
from mdagent.subagents.subagent_setup import SubAgentInitializer, SubAgentSettings


@pytest.fixture(scope="session", autouse=True)
def set_env():
    load_dotenv()


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

    mm_path_id = MemoryManager(get_registry("raw", False), run_id="TESTRUNN")
    assert mm_path_id.run_id == "TESTRUNN"

    assert os.path.exists(f"{mm_path_id.dir_name}/memories/memory_details")
    assert os.path.exists(f"{mm_path_id.dir_name}/memories")


def test_write_to_and_retrieve_from_history(memory_manager):
    input_dict = {
        "prompt": "prompt_",
        "attempt_number": 1,
        "code": "code_",
        "output": "output_",
        "critique": "critique_",
        "success": True,
    }
    memory_manager._write_history_iterator(**input_dict)
    assert os.path.exists(memory_manager.memory_path)
    with open(memory_manager.memory_path, "r") as f:
        data = json.load(f)
    assert data["prompt__1"] == input_dict

    memory = memory_manager.retrieve_recent_memory_iterator(last_only=True)
    assert str(memory) == str(input_dict)


def test_pull_memory_summary(memory_manager, get_registry):
    # write fake summary
    fake_summaries = {"TESTRUNN": "fake_summary"}
    with open(memory_manager.memory_summary_path, "w") as f:
        f.write(json.dumps(fake_summaries))
    assert memory_manager.pull_memory_summary("NOTARUNN") is None
    assert memory_manager.run_id_mem is None

    mm_mem = MemoryManager(get_registry("raw", False), run_id="TESTRUNN")
    assert mm_mem.pull_memory_summary() == "fake_summary"

    with open(memory_manager.agent_summary_path, "w") as f:
        f.write(json.dumps(fake_summaries))
    mm_mem = MemoryManager(get_registry("raw", False), run_id="TESTRUNN")
    assert mm_mem.pull_agent_summary_from_mem() == "fake_summary"
    assert mm_mem.run_id_mem == "fake_summary"
