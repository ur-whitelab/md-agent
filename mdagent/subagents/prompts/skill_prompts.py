SKILL_PREFIX = """
You are a helpful assistant that writes a description of the given python code.

I will give you the python function code.

You must follow the following criteria:
1) Try to summarize the function description in no more than 6 sentences.
2) The description should contain expected inputs and outputs and explain
clearly what the code does.
3) Description should be in docstrings with up to 80 characters per line.
"""
SKILL_FORMAT = '''
You should only respond with docstrings:

RESPONSE FORMAT:
""" this should be tool description in format of docstrings """
'''
SKILL_PROMPT = """
INPUT:
python code: {code}
"""
SKILL_INPUT_VARIABLES = ["code"]


class SkillPrompts:
    PREFIX = SKILL_PREFIX
    FORMAT = SKILL_FORMAT
    PROMPT = SKILL_PROMPT
    INPUT_VARS = SKILL_INPUT_VARIABLES
