import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mdagent.subagents.agents.action import Action
from mdagent.subagents.subagent_fxns import Iterator
from mdagent.utils import PathRegistry


@pytest.fixture
def path_registry():
    return PathRegistry()


@pytest.fixture
def action(path_registry):
    return Action(path_registry)


@pytest.fixture
def iterator(path_registry):
    return Iterator(path_registry)


def test_exec_code(action):
    successful_code = "print('Hello, World!')"
    success, _ = action._exec_code(successful_code)
    # assert success
    assert success is True
    error_code = "raise ValueError('Test Error')"
    success, _ = action._exec_code(error_code)
    # assert failure
    assert success is False


def test_extract_code(action):
    # test1 is valid code
    sample_output = (
        "Here's some code:\n```"
        "\ndef sample_function():\n    return 'Hello, World!'\n```"
    )
    # Call the _extract_code function with the sample output
    code, fxn_name = action._extract_code(sample_output)

    # Assert that the code and function name are correctly extracted
    expected_code = "def sample_function():\n    return 'Hello, World!'"
    assert code == expected_code
    assert fxn_name == "sample_function"

    # test2 is two types of invalid code
    no_code = "text without code."
    code_1, fxn_name_1 = action._extract_code(no_code)
    no_fxn = "Here's some code:\n```python\nx = 10\ny = 20\n```"
    code_2, fxn_name_2 = action._extract_code(no_fxn)
    assert code_2 == "x = 10\ny = 20"
    assert code_1 is None
    assert fxn_name_1 is None
    assert fxn_name_2 is None


def test_add_to_history(iterator):
    iterator.path_registry = MagicMock()
    iterator.path_registry.list_path_names.return_value = ["file1.txt", "file2.txt"]

    existing_history = []
    iter = 1
    task = "Sample Task"
    code_history = "print('Hello, World!')"
    output_history = "Hello, World!"
    critique = "Good code"
    suggestions = "None"

    updated_history = iterator._add_to_history(
        existing_history,
        iter,
        task,
        code_history,
        output_history,
        critique,
        suggestions,
    )

    assert len(updated_history) == 1
    history_item = json.loads(updated_history[0])
    assert history_item["iteration"] == iter
    assert history_item["task"] == task
    assert history_item["code"] == code_history
    assert history_item["output"] == output_history
    assert history_item["files"] == ["file1.txt", "file2.txt"]
    assert history_item["critique"] == critique
    assert history_item["suggestions"] == suggestions


def test_pull_information(iterator):
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="line1\nline2\nline3")):
            iterator.skill = MagicMock()
            iterator.skill.get_skills.return_value = ["skill1", "skill2"]
            iterator.path_registry = MagicMock()
            iterator.path_registry.list_path_names.return_value = ["file1", "file2"]
            iterator.current_tools = {"tool1": "config1"}
            iterator.all_tools_string = "all_tools_string"
            info = iterator._pull_information()

            assert info["recent_history"] == "line3"
            assert info["full_history"] == "line1\nline2\nline3"
            assert info["skills"] == json.dumps(["skill1", "skill2"])
            assert info["files"] == json.dumps(["file1", "file2"])
            assert info["current_tools"] == json.dumps({"tool1": "config1"})
            assert info["all_tools"] == "all_tools_string"
