critic_format = """
You should only respond in JSON format as described below:
{
    "success": boolean,
    "critique": "critique",
}
Ensure the response can be parsed by Python `json.loads`, 
e.g.: no trailing commas, no single quotes, etc.
"""

critic_prefix = """
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
"""

critic_prompt = """
INPUT: 
files: {files},
code: {code},
output: {code_output},
task: {task},
context: {context},
additional_information: {additional_information}
"""