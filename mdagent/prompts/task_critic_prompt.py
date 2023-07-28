critic_format = """
You should only respond in JSON format as described below:
{
    "success": boolean,
    "critique": "critique",
}
Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
"""

critic_prefix = """
You are an assistant that assesses my progress on my molecular dynamics project and provides useful guidance. 

You are required to evaluate if I have met the task requirements. 
If I have met or exceeded the requirements, your response should be 'yes'. 
However, if I have failed to meet the requirements, your response should be 'no', 
accompanied by critique to help me improve. 

I will give you the following information.


Files: All files created in the current directory, along with descriptions
Task: The objective I need to accomplish
Context: The context of the task
"""

critic_prompt = """
INPUT: 
files: {files},
output: {code_output},
task: {task},
contex: {context}
"""