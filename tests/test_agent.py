import pytest

from mdagent.subagents.agents.action import Action


@pytest.fixture
def action():
    return Action()


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
    sample_output = "Here's some code:\n```\ndef sample_function():\n    return 'Hello, World!'\n```"
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
