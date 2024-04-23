from unittest.mock import MagicMock, mock_open, patch

import pytest

from mdagent.mainagent.evaluate import Evaluator


@pytest.fixture
def evaluator():
    with patch("os.makedirs"):
        yield Evaluator()


@pytest.fixture
def mock_os_makedirs():
    with patch("os.makedirs", MagicMock()) as mock:
        yield mock


@pytest.fixture
def mock_open_json():
    with patch("builtins.open", mock_open(read_data='[{"key": "value"}]')) as mock:
        yield mock


@pytest.fixture
def mock_json_load():
    with patch("json.load", return_value=[{"key": "value"}]) as mock:
        yield mock


@pytest.fixture
def mock_json_dump():
    with patch("json.dump", MagicMock()) as mock:
        yield mock


@pytest.fixture
def mock_os_path_exists():
    with patch("os.path.exists", return_value=True) as mock:
        yield mock


@pytest.fixture
def mock_agent(tmp_path):
    mock_action = MagicMock()
    mock_action.tool = "some_tool"
    mock_action.tool_input = "some_input"
    agent = MagicMock()
    agent.iter.return_value = iter(
        [
            {"intermediate_step": [(mock_action, "Succeeded. some obervation.")]},
            {"output": "Succeeded. Some final answer."},
        ]
    )
    agent.ckpt_dir = tmp_path / "fake_dir"
    agent.llm.model_name = "test_model"
    agent.tools_llm.model_name = "some_tool_model"
    agent.subagents_settings.subagents_model = "test_subagent_model"
    agent.agent_type = "test_agent_type"
    agent.subagents_settings.resume = False
    agent.subagents_settings.curriculum = True
    agent.use_memory = False
    agent.run_id = "test_run_id"
    return agent


@patch("mdagent.mainagent.evaluate.MDAgent")
def test_create_agent(mock_mdagent, evaluator):
    agent_params = {"model_name": "test_model"}
    evaluator.create_agent(agent_params)
    mock_mdagent.assert_called_once_with(**agent_params)


def test_reset(evaluator):
    evaluator.evaluations = ["dummy"]
    evaluator.reset()
    assert evaluator.evaluations == []


def test_save(evaluator, mock_open_json, mock_json_dump):
    evaluator.evaluations = [{"test_key": "test_value"}]
    evaluator.save("test_file")
    mock_open_json.assert_called()
    mock_json_dump.assert_called()


def test_load(evaluator, mock_open_json, mock_json_load, mock_os_path_exists):
    filename = "dummy_data.json"
    evaluator.load(filename)
    mock_os_path_exists.assert_called_once_with(filename)
    mock_open_json.assert_called_once_with(filename, "r")
    mock_json_load.assert_called_once()
    assert evaluator.evaluations == [{"key": "value"}]


def test_evaluate_all_steps(evaluator, mock_agent, mock_os_makedirs, mock_open_json):
    user_prompt = "Test prompt"
    result = evaluator._evaluate_all_steps(mock_agent, user_prompt)
    assert result["prompt_success"] is True, "The prompt should be marked as succeeded."


def test_evaluate_all_steps_contents(
    evaluator, mock_agent, mock_os_makedirs, mock_open_json, mock_json_dump
):
    user_prompt = "Test some prompt"
    evaluator._evaluate_all_steps(mock_agent, user_prompt)
    assert mock_json_dump.call_count == 1
    args, kwargs = mock_json_dump.call_args
    data_to_dump = args[0]
    assert data_to_dump["prompt_success"] is True
    assert "Step 1" in data_to_dump["tools_details"]
    assert data_to_dump["tools_details"]["Step 1"]["status_complete"] is True
    assert data_to_dump["total_steps"] == 1


def test_run_and_evaluate(evaluator, mock_os_makedirs, mock_open_json):
    with patch(
        "mdagent.mainagent.evaluate.Evaluator._evaluate_all_steps"
    ) as mock_evaluate_all_steps:
        mock_evaluate_all_steps.side_effect = [
            {"prompt_success": True},
            Exception("Test error"),
        ]
        prompts = ["Prompt 1", "Prompt 2"]
        evaluator.run_and_evaluate(prompts)
        assert len(evaluator.evaluations) == 2
        assert evaluator.evaluations[0]["execution_success"] is True
        assert evaluator.evaluations[1]["execution_success"] is False
        assert "Test error" in evaluator.evaluations[1]["error_msg"]


@patch("pandas.DataFrame.to_json", MagicMock())
def test_create_table(evaluator):
    evaluator.evaluations = [
        {
            "execution_success": True,
            "total_steps": 1,
            "failed_steps": 0,
            "prompt_success": True,
            "total_time_seconds": "10.0",
        },
        {
            "execution_success": True,
            "total_steps": 2,
            "failed_steps": 1,
            "prompt_success": False,
            "total_time_seconds": "20.0",
        },
    ]
    df = evaluator.create_table()
    assert len(df) == 2
