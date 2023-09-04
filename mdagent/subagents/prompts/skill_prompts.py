SKILL_PREFIX = """
You are a helpful assistant that writes a description of the given function 
written in python code.

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

SKILL_FORMAT = """
You should only respond in the format as described below:

RESPONSE FORMAT:
Function name: this should be Python function name
Tool name: this should be Python name for class. 
Tool description: this should be tool description in format of docstrings.
"""

SKILL_PROMPT = """
INPUTS:
python code: {code}
"""

class SkillPrompts:
    PREFIX = SKILL_PREFIX
    FORMAT = SKILL_FORMAT
    PROMPT = SKILL_PROMPT
