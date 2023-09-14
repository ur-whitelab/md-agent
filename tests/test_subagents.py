import json
import re

import pytest
from dotenv import load_dotenv

from mdagent.subagents.subagent_setup import SubAgentInitializer, SubAgentSettings


@pytest.fixture(scope="session", autouse=True)
def set_env():
    load_dotenv()


def test_subagent_setup():
    settings = SubAgentSettings(path_registry=None)
    initializer = SubAgentInitializer(settings)
    subagents = initializer.create_iteration_agents()
    action_agent = subagents["action"]
    code_critic_agent = subagents["code_critic"]
    curriculum_agent = subagents["refining_curriculum"]
    skill_agent = subagents["skill"]
    task_critic_agent = subagents["task_critic"]
    assert action_agent is not None
    assert code_critic_agent is not None
    assert curriculum_agent is not None
    assert skill_agent is not None
    assert task_critic_agent is not None


@pytest.fixture(scope="module")
def action_agent():
    settings = SubAgentSettings(path_registry=None)
    initializer = SubAgentInitializer(settings)
    subagents = initializer.create_iteration_agents()
    return subagents["action"]


def test_extract_code(action_agent):
    # Test Case 1: Check the extraction of code.
    output1 = "Some text here.\nCode:\n```\nprint('Hello World')\n```\nSome other text."
    assert action_agent._extract_code(output1) == "print('Hello World')"

    # Test Case 2: Check when there's no code block.
    output2 = "Some text here. No code block in this text."
    assert action_agent._extract_code(output2) is None

    # Test Case 3: Check extraction with multiple lines of code.
    output3 = """Some text here.\nCode:
                \n```\ndef hello():\n
                print('Hello World')\n```\n
                Some other text."""
    expected_output3 = "def hello():\n    print('Hello World')"
    assert action_agent._extract_code(output3) == expected_output3

    # Test Case 4: Check when there's code block but no 'Code:' prefix.
    output4 = "Some text here.\n```\nprint('Hello World')\n```\nSome other text."
    assert action_agent._extract_code(output4) is None

    # Test Case 5: Check for empty code block.
    output5 = "Some text here.\nCode:\n```\n\n```\nSome other text."
    print(action_agent._extract_code(output5))
    assert action_agent._extract_code(output5) is None


def test_exec_valid_code(action_agent):
    # Scenario 1: Valid JSON input that contains valid Python code
    code = json.dumps({"code": "print('Hello World!')"})
    success, output = action_agent._exec_code(code)
    assert success is True
    assert "Hello World!" in output


def test_exec_invalid_code(action_agent):
    # Scenario 2: Valid JSON input but contains invalid Python code
    code = json.dumps(
        {"code": "print(Hello World!)"}
    )  # Missing quotes around Hello World!
    success, output = action_agent._exec_code(code)
    assert success is False
    assert "invalid syntax" in output


def test_exec_invalid_json(action_agent):
    # Scenario 3: Invalid JSON input
    code = "This is not a JSON string"
    with pytest.raises(json.JSONDecodeError):
        action_agent._exec_code(code)


@pytest.fixture(scope="module")
def skill_agent():
    settings = SubAgentSettings(path_registry=None)
    initializer = SubAgentInitializer(settings)
    subagents = initializer.create_iteration_agents()
    return subagents["skill"]


def test_function_description(skill_agent):
    description = skill_agent.llm_step1({"code": "print('Hello World!')"})["text"]
    # Check Function name
    print(description)
    match = re.search(r"Function name: (\w+)", description)
    assert match is not None, "Function name section missing or incorrectly formatted"
    function_name = match.group(1)
    assert (
        function_name.isidentifier()
    ), f"Function name '{function_name}' is not a valid Python identifier"

    # Check Tool name
    match = re.search(r"Tool name: (\w+)", description)
    assert match is not None, "Tool name section missing or incorrectly formatted"
    tool_name = match.group(1)
    assert (
        tool_name.isidentifier()
    ), f"Tool name '{tool_name}' is not a valid Python identifier"

    # Check Tool description
    match = re.search(r"Tool description: (.+)", description, re.DOTALL)
    assert (
        match is not None
    ), "Tool description section missing or incorrectly formatted"
    tool_description = match.group(1).strip()
    assert len(tool_description) > 0, "Tool description is empty"

    return True
