from typing import Optional

from langchain.tools import BaseTool
from . import PathRegistry, Iterator

class GetNewTool(BaseTool):
    name = "GetNewTool"
    description = """
        This tool is used to create a new tool.
        Given a description of the tool needed,
        it will write and test tools.
        If this tool hits maximum iterations without suceeding
        to create a tool, it will return a failure. If you
        receive a failure, you can try again with a different
        input description. If you receive a success, you will
        recieve the tool name, description, and input type.
        You can then use the tool in subsequent steps.
        Follow this format for your input:
        Tool: [tool description, input and output should be 1 string each]
        Context: [the full user prompt from the beginning, 1 string]
    """

    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, query: str) -> str:
        """use the tool."""

        # check formatting
        try:
            lower_input = query.lower()
            task = lower_input.split("tool:", 1)[-1].split("context:", 1)[0].strip()
            context = lower_input.split("context:", 1)[-1].strip()
            if any(item in (None, "") for item in (task, context)):
                raise ValueError("incorrect input")
        except Exception:
            return "Incorrect input format. Please try again."

        try:
            # run iterator
            iterator = Iterator(self.path_registry)
            success, history = iterator._run_iteration(1, task, context)
            if success is True:
                # we should get tool name,
                # description, and
                # input type from skills manager
                return """Tool created successfully.
            You can now use the tool in subsequent steps."""
            else:
                return "Tool creation failed. Try again with a different description."
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")