STEP1_PREFIX = """
You are a helpful assistant that writes a description of the given python code.

I will give you the python function code.

You must follow the following criteria:
1) the function name must be one word string, following PEP8 rules for Python
function names.
2) the tool name must be one word string similar to function name except that
it follows PEP8 rules for Python class names (i.e. CamelCase naming).
2) Try to summarize the function description in no more than 6 sentences.
3) The description should contain expected inputs and outputs and explain
clearly what the code does.
4) Description should be in docstrings with up to 80 characters per line.
5) Function name, tool name, and description must be in separate lines.
"""

STEP1_FORMAT = """
You should only respond in the format as described below:

RESPONSE FORMAT:
Function name: this should be Python function name
Tool name: this should be Python name for class.
Tool description: this should be tool description in format of docstrings.
"""

STEP1_PROMPT = """
INPUT:
python code: {code}
"""

STEP1_INPUT_VARIABLES = ["code"]

STEP2_PREFIX = '''
You are a helpful assistant that writes a full code, the contents in a
complete python script for the given basic code & other essential pieces of
information.

I will give you the python code, function name, tool name, and description.

You must follow the following criteria:
1) the given code must be wrapped in a Python function, naming it with
the given function name.
2) below the function code, create a class as BaseTool object, naming it
with the given tool name. Basetool object must has the following exact
functions: __init__, _run, and _async
3) the full code must contain all imports.
4) the full code must follow closely to the following example format
as much as possible, <FIELDS> tells you where you can place the given
inputs.

Example format of the full code:
`from typing import Optional

<ANY OTHER IMPORTS HERE>
from langchain.tools import BaseTool

from .registry import PathRegistry


def <FXN_NAME>(<FXN_INPUTS>, PathRegistry):
    <CODE HERE>

class <TOOL_NAME>(BaseTool):
    name = <TOOL_NAME>
    description = <DESCRIPTION>
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        super().__init__()
        self.path_registry = path_registry

    def _run(self, <FXN_INPUTS>: str) -> str:
        """Use the tool."""
        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            output = <FXN_NAME>(<FXN_INPUTS>, self.path_registry)
            if output is None:
                return "This tool fails to produce the expected output"
            else:
                return f"This tool completed its task. Output:" + str(output)
        except Exception as e:
            return "Something went wrong" + str(e)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Name2PDB does not support async")
`
'''

STEP2_FORMAT = """
You must only respond in the format as described below:

RESPONSE FORMAT:
Full Code: this should be complete contents in a Python file.
"""

STEP2_PROMPT = """
INPUTS:
code: {code},
fxn_name: {fxn_name},
tool_name: {tool_name},
description: {description},
"""

STEP2_INPUT_VARIABLES = ["code", "fxn_name", "tool_name", "description"]


class SkillStep1Prompts:
    PREFIX = STEP1_PREFIX
    FORMAT = STEP1_FORMAT
    PROMPT = STEP1_PROMPT
    INPUT_VARS = STEP1_INPUT_VARIABLES


class SkillStep2Prompts:
    PREFIX = STEP2_PREFIX
    FORMAT = STEP2_FORMAT
    PROMPT = STEP2_PROMPT
    INPUT_VARS = STEP2_INPUT_VARIABLES
