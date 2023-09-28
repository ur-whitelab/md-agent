import io
import os
import sys
from typing import Optional

from langchain.tools import BaseTool

from mdagent.subagents import Iterator, SubAgentInitializer, SubAgentSettings
from mdagent.utils import PathRegistry


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
        Prompt: [the full user prompt from the beginning, 1 string]
    """
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]

    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        subagent_settings: Optional[SubAgentSettings],
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, query: str) -> str:
        """use the tool."""

        # check formatting
        try:
            lower_input = query.lower()
            task = lower_input.split("Tool:", 1)[-1].split("Prompt:", 1)[0].strip()
            original_prompt = lower_input.split("Prompt:", 1)[-1].strip()
            if any(item in (None, "") for item in (task, original_prompt)):
                raise ValueError("incorrect input")
        except Exception:
            return "Incorrect input format. Please try again."

        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            if self.subagent_settings is None:
                return "Settings for subagents yet to be defined"

            # run iterator
            newcode_iterator = Iterator(self.path_registry, self.subagent_settings)
            tool_name = newcode_iterator.run(task, original_prompt)
            if tool_name:
                return f"""Tool created successfully: {tool_name}
                You can now use the tool in subsequent steps."""
            else:
                return "The 'GetNewTool' tool failed to build a new tool."
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")


def execute_skill_code(tool_name, skill_agent, path_registry):
    skills = skill_agent.get_skills()
    code = skills.get(tool_name, {}).get("code", None)

    if not code:
        raise ValueError(
            f"Code for {tool_name} not found. Make sure to use correct tool name."
        )

    # capture initial state
    initial_files = set(os.listdir("."))
    initial_registry = path_registry.list_path_names()

    # Redirect stdout and stderr to capture the output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = captured_stdout = sys.stderr = io.StringIO()
    exec_context = {**globals(), **locals()}  # to allow for imports

    try:
        exec(code, exec_context, exec_context)
        output = captured_stdout.getvalue()
    except Exception as e:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        error_type = type(e).__name__
        raise type(e)(f"Error executing code for {tool_name}: {error_type} - {e}")
    finally:
        # Ensure that stdout and stderr are always restored
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    # capture final state
    new_files = list(set(os.listdir(".")) - initial_files)
    new_registry = list(set(path_registry.list_path_names()) - set(initial_registry))

    success_message = "Successfully executed code."
    files_message = f"New Files Created: {', '.join(new_files)}"
    registry_message = f"Files added to Path Registry: {', '.join(new_registry)}"
    output_message = f"Code Output: {output}"

    return "\n".join([success_message, files_message, registry_message, output_message])


class ExecuteSkillCode(BaseTool):
    name = "ExecuteSkill"
    description = """Executes the code for a new tool that has been
    recently added during the current iteration.

    Inputs:
    - tool_name: a string representing the name of the tool for which
        the code needs to be executed.

    Outputs:
    - If the code for the specified tool is found, it is executed
    successfully and a success message is returned.
    - If there is an error while executing the code, an error message
    along with the error details is returned.
    - If the code for the specified tool is not found, a message
    indicating that the code was not found is returned.

    The function retrieves the skills from the skill agent and then
    retrieves the code for the specified tool from the skills dictionary.
    If the code is found, it is executed using the exec() function. If
    there is an error during execution, the error message is returned.
    If the code is not found, a message indicating that the code was not
    found is returned."""

    """Execute the code for a new tool that has been recently added
    during the current iteration. The function retrieves the code for
    the specified tool from skill library.
    If the code is not found, a ValueError is raised.
    If an exception occurs during code execution, the exception is raised.

    Inputs:
    - tool_name: a string representing the name of the tool for which
        the code needs to be executed.

    Returns:
    - the function returns a string containing a success message, a list
    of new files created, a list of files added to the path registry,
    and the output of the executed code.
    """
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]

    def __init__(
        self,
        path_registry: Optional[PathRegistry] = None,
        subagent_settings: Optional[SubAgentSettings] = None,
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, query: str) -> str:
        """use the tool"""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            agent_initializer = SubAgentInitializer(self.subagent_settings)
            skill_agent = agent_initializer.create_skill_agent(resume=True)
            if skill_agent is None:
                return "Agent for this tool not initialized"
            code_result = execute_skill_code(query, skill_agent, self.path_registry)
            return code_result
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("This tool does not support async")
