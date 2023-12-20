import json
from unittest.mock import MagicMock

import pytest

from mdagent.subagents.agents.action import Action
from mdagent.subagents.subagent_fxns import Iterator


@pytest.fixture
def action():
    return Action()


@pytest.fixture
def iterator():
    return Iterator()


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

    # Define sample inputs
    existing_history = []
    iter = 1
    task = "Sample Task"
    code_history = "print('Hello, World!')"
    output_history = "Hello, World!"
    critique = "Good code"
    suggestions = "None"

    # Call the _add_to_history function with the sample inputs
    updated_history = iterator._add_to_history(
        existing_history,
        iter,
        task,
        code_history,
        output_history,
        critique,
        suggestions,
    )

    # Assert that the history has one new item
    assert len(updated_history) == 1

    # Convert the added history item back to a dictionary for verification
    history_item = json.loads(updated_history[0])

    # Assert that all fields are correctly added to the history item
    assert history_item["iteration"] == iter
    assert history_item["task"] == task
    assert history_item["code"] == code_history
    assert history_item["output"] == output_history
    assert history_item["files"] == ["file1.txt", "file2.txt"]
    assert history_item["critique"] == critique
    assert history_item["suggestions"] == suggestions
