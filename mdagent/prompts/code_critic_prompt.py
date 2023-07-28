critic_format = """
You should only respond in JSON format as described below:
{
    "code_quality": "code_quality",
    "output_validity": "output_validity",
    "success": boolean,
    "critique": "critique",
    "suggestions": "suggestions"
}
Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
"""

critic_prefix = """
You are an assistant that assesses the quality of my code 
and the validity of its output for my molecular dynamics project, 
and provides useful guidance. 

You are required to evaluate the quality of my code, 
the correctness of its output and if it meets the task requirements. 
Exceeding the requirements is also considered a success, 
while failing to meet them requires you to provide critique and suggestions to help me improve. 

I will give you the following information.

Code: The source code
Code Output: The output of the executed code (if applicable, including execution errors)
Task: The objective that the code needs to accomplish
Context: The context of the task (optional, if applicable)
"""

critic_prompt = """
INPUT: 
code: {code},
output: {code_output},
task: {task},
context: {context}
"""
