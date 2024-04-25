import pytest
from pydantic import BaseModel

from mdagent.utils import validate_func_args, validate_tool_args


@pytest.fixture()
def test_function():
    @validate_func_args()
    def test_function(name, age):
        return "All good"

    return test_function


def test_validate_func_args(test_function):
    assert test_function(name="John", age=30) == "All good"
    assert "Did you mean 'name'?" in test_function(wrongname="John", age=30)
    assert "will be ignored." in test_function(name="John", age=30, wrongarg="test")


@pytest.fixture()
def argument_schema():
    class ArgsSchema(BaseModel):
        name: str
        age: int

    return ArgsSchema


@pytest.fixture()
def test_tool(argument_schema):
    class TestTool:
        args_schema = argument_schema

        @validate_tool_args(args_schema=args_schema)
        def _run(self, **kwargs):
            return "All good"

    return TestTool


def test_validate_tool_args(test_tool):
    assert test_tool()._run(name="John", age=30) == "All good"
    assert "Did you mean 'name'?" in test_tool()._run(wrongname="John", age=30)
    assert "will be ignored." in test_tool()._run(name="John", age=30, wrongarg="test")
