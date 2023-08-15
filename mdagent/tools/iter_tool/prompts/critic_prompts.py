# code critic
code_critic_format = """
You should only respond in JSON format as described below:
{
    "code_quality": "code_quality",
    "output_validity": "output_validity",
    "success": boolean,
    "critique": "critique",
    "suggestions": "suggestions"
}
Ensure the response can be parsed by Python `json.loads`,
e.g.: no trailing commas, no single quotes, etc.
"""

code_critic_prefix = """
You are an assistant that assesses the quality of my code
and the validity of its output for my molecular dynamics project,
and provides useful guidance.

You are required to evaluate the quality of my code,
the correctness of its output and if it meets the task requirements.
Exceeding the requirements is also considered a success,
while failing to meet them requires you to
provide critique and suggestions to help me improve.

I will give you the following information.

Code: The source code
Code Output: The output of the executed code (if applicable, including execution errors)
Task: The objective that the code needs to accomplish
Context: The context of the task (optional, if applicable)

You should only respond in the format as described below:

RESPONSE FORMAT:
{code_critic_format}

"""

code_critic_prompt = """
INPUT:
code: {code},
output: {code_output},
task: {task},
context: {context}
"""

# task critic
task_critic_format = """
You should only respond in JSON format as described below:
{
    "success": boolean,
    "critique": "critique",
}
Ensure the response can be parsed by Python `json.loads`,
e.g.: no trailing commas, no single quotes, etc.
"""

task_critic_prefix = """
You are an assistant that assesses my progress on my molecular
dynamics project and provides useful guidance.

You are required to evaluate if I have succesfully met
all of the task requirements.
If I have met or exceeded the requirements,
your response should be True.
However, if I have failed to meet the requirements,
your response should be False,
accompanied by critique to help me improve.

I will give you the following information.


Files: All files created in the current directory,
along with descriptions when applicable.
Code: The code I have written
Output: The execution output of the code I have written
Task: The objective I need to accomplish
Context: The context of the task
Additional Information: Any additional information, in case you cause an error.

You should only respond in the format as described below:

RESPONSE FORMAT:
{code_format}
"""

task_critic_prompt = """
INPUT:
files: {files},
code: {code},
output: {code_output},
task: {task},
context: {context},
additional_information: {additional_information}
"""

# action critic --> not currently used.. do we need this?
action_critic_format = """
You should only respond in JSON format as described below:
{
    "incorrect_code": boolean,
    "incorrect_output": boolean,
    "no_skill: boolean,
    "combine_skill": boolean,
    "wrong_choice": boolean,
    "suggestions": "suggestions"
}
Ensure the response can be parsed by Python `json.loads`,
e.g.: no trailing commas, no single quotes, etc.
"""

action_critic_prefix = """
You are an assistant that assesses the quality of my code
and provides useful guidance on why my code failed to execute.
I will give you the following information:

Task: The objective that the code needs to accomplish
Skills (list): All skills available to me
Output: The output of the executed code (if applicable, including execution errors)

You are required to tell me why my code did not execute by answering
True or False to the following:

1. The code is syntactically or semantically incorrect
2. The input is invalid
3. No skill in the skill list can execute the code
4. Skills from the skill list can execute the code, if combined
5. The chosen skill is not the best skill to execute the code

You must also give me suggestions on how to fix the problem.

You should only respond in the format as described below:

RESPONSE FORMAT:
{code_format}

If you get an error message in the input variable called errors,
please adjust your response accordingly.
"""

action_critic_prompt = """
INPUT:
task: {task}
skills: {skills}
output: {output}
errors: {errors}
"""
