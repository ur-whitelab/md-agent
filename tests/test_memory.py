import json
import os

import pytest

from mdagent.agent.agent import MDAgent
from mdagent.agent.memory import MemoryManager


@pytest.fixture
def memory_manager(get_registry):
    return MemoryManager(get_registry("raw", False))


def test_mdagent_memory():
    mdagent_memory = MDAgent(use_memory=True)
    mdagent_no_memory = MDAgent(use_memory=False)
    assert mdagent_memory.use_memory is True
    assert mdagent_no_memory.use_memory is False

    mdagent_memory = MDAgent(use_memory=True, run_id="TESTRUNN")
    assert mdagent_memory.run_id == "TESTRUNN"

    mdagent_memory = MDAgent(use_memory=True, run_id="")
    assert mdagent_memory.run_id


def test_memory_init(memory_manager, get_registry):
    assert memory_manager is not None
    assert memory_manager.run_id is not None
    assert len(memory_manager.run_id) == 8

    mm_path_id = MemoryManager(get_registry("raw", False), run_id="TESTRUNN")
    assert mm_path_id.run_id == "TESTRUNN"


def test_force_clear_mem(monkeypatch):
    dummy_test_dir = "ckpt_test"

    mdagent = MDAgent(ckpt_dir=dummy_test_dir)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    print(mdagent.path_registry.ckpt_dir)
    print(mdagent.path_registry.json_file_path)
    print(os.path.dirname(mdagent.path_registry.ckpt_dir))
    mdagent.force_clear_mem(all=False)
    assert not os.path.exists(mdagent.path_registry.ckpt_dir)
    assert not os.path.exists(mdagent.path_registry.json_file_path)
    assert os.path.exists(
        os.path.basename(os.path.dirname(mdagent.path_registry.ckpt_dir))
    )

    mdagent = MDAgent(ckpt_dir=dummy_test_dir)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    mdagent.force_clear_mem(all=True)
    print(mdagent.path_registry.ckpt_dir)
    print(mdagent.path_registry.json_file_path)
    print(os.path.dirname(mdagent.path_registry.ckpt_dir))
    assert not os.path.exists(mdagent.path_registry.ckpt_dir)
    assert not os.path.exists(mdagent.path_registry.json_file_path)
    assert not os.path.exists(os.path.dirname(mdagent.path_registry.ckpt_dir))


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
