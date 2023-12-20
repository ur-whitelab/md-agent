import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mdagent.subagents.agents.action import Action
from mdagent.subagents.agents.skill import SkillManager
from mdagent.subagents.subagent_fxns import Iterator
from mdagent.subagents.subagent_setup import SubAgentSettings
from mdagent.utils import PathRegistry


@pytest.fixture
def path_registry():
    return PathRegistry()


@pytest.fixture
def skill_manager(path_registry):
    return SkillManager(path_registry=path_registry)


@pytest.fixture
def action(path_registry):
    return Action(path_registry)


@pytest.fixture
def iterator(path_registry):
    settings = SubAgentSettings(path_registry=None)
    return Iterator(path_registry=path_registry, subagent_settings=settings)


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
        "Here's some code. \nCode:\n```python\n"
        "def sample_function():\n    return 'Hello, World!'\n```"
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
    no_fxn = "Code:\n```python\nx = 10\ny = 20\n```"
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


def test_add_new_tool(skill_manager):
    # mock exec so tests are independent
    def mock_exec(code, namespace):
        def sample_function():
            """Sample Docstring"""
            return "Hello, World!"

        namespace["sample_function"] = sample_function

    fxn_name = "sample_function"
    code = (
        "def sample_function():\n    '''Sample Docstring'''\n    return 'Hello, World!'"
    )
    skill_manager._generate_tool_description = MagicMock(
        return_value="Generated Description"
    )
    skill_manager.update_skill_library = MagicMock()

    with patch("builtins.exec", side_effect=mock_exec):
        result = skill_manager.add_new_tool(fxn_name, code, new_description=False)
        assert result == fxn_name
        assert skill_manager.update_skill_library.call_args[0][0].__name__ == fxn_name
        assert skill_manager.update_skill_library.call_args[0][1] == code
        assert skill_manager.update_skill_library.call_args[0][2] == "Sample Docstring"


def test_execute_skill_function(skill_manager):
    path_registry = MagicMock()
    path_registry.list_path_names.return_value = ["path1", "path2"]
    skill_manager.skills = {
        "sample_tool": {"code": "def sample_tool(arg1, arg2):\n    return arg1 + arg2"}
    }
    with patch("os.listdir", return_value=["file1", "file2"]):
        skill_manager._check_arguments = MagicMock()
        message = skill_manager.execute_skill_function(
            "sample_tool", path_registry, arg1=5, arg2=3
        )

    assert "Successfully executed code." in message
    assert "Code Output: 8" in message
    skill_manager.skills = {}
    with pytest.raises(ValueError) as excinfo:
        skill_manager.execute_skill_function("nonexistent_tool", path_registry)
    assert "Code for nonexistent_tool not found" in str(excinfo.value)


def test_check_arguments_success(skill_manager):
    skill_manager.skills = {
        "sample_tool": {"arguments": [{"name": "arg1"}, {"name": "arg2"}]}
    }
    try:
        skill_manager._check_arguments("sample_tool", arg1=5, arg2=10)
    except ValueError:
        pytest.fail("ValueError raised unexpectedly")
    with pytest.raises(ValueError) as excinfo:
        skill_manager._check_arguments("sample_tool", arg1=5)
    assert "Missing arguments" in str(excinfo.value)


def test_retrieve_skills(skill_manager):
    skill_manager.vectordb = MagicMock()
    skill_manager.vectordb._collection.count.return_value = 10
    skill_manager.vectordb.similarity_search_with_score.return_value = [
        (MagicMock(metadata={"name": "skill1"}), 0.9),
        (MagicMock(metadata={"name": "skill2"}), 0.8),
    ]

    skill_manager.skills = {
        "skill1": {"code": "code for skill1"},
        "skill2": {"code": "code for skill2"},
    }
    skill_manager.retrieval_top_k = 5

    retrieved_skills = skill_manager.retrieve_skills("query")

    assert len(retrieved_skills) == 2
    assert "skill1" in retrieved_skills
    assert "skill2" in retrieved_skills
    assert retrieved_skills["skill1"] == "code for skill1"
    assert retrieved_skills["skill2"] == "code for skill2"


def test_update_skill_library(skill_manager):
    # Mock the dependencies
    skill_manager.vectordb = MagicMock()
    skill_manager.path_registry = MagicMock()
    skill_manager.dir_name = "/mock_dir"

    with patch("os.listdir", return_value=[]), patch(
        "os.path.exists", return_value=False
    ), patch("builtins.open", new_callable=MagicMock()) as mock_open:
        sample_function = MagicMock()
        sample_function.__name__ = "test_function"
        code_script = "def test_function(): pass"
        description = "Test function description"
        arguments = []
        skill_manager.update_skill_library(
            sample_function, code_script, description, arguments
        )
        mock_open.assert_any_call("/mock_dir/code/test_function.py", "w")
        mock_open.assert_any_call("/mock_dir/description/test_function.txt", "w")
        mock_open.assert_any_call("/mock_dir/skills.json", "w")

        skill_manager.vectordb.add_texts.assert_called_once_with(
            texts=[description],
            ids=["test_function"],
            metadatas=[{"name": "test_function"}],
        )
        skill_manager.vectordb.persist.assert_called_once()
        skill_manager.path_registry.map_path.assert_called_once_with(
            name="test_function",
            path="/mock_dir/code/test_function.py",
            description="Code for new tool test_function",
        )
