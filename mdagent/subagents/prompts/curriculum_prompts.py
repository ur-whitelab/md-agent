from langchain.prompts import PromptTemplate

CURRICULUM_PROMPT = """
You are an expert molecular dynamics scientist and your role is to set
a curriculum plan with subtasks to complete the final task specified by me.
This is similar to molecular dynamics workflow planning that includes
determining what kind of preprocessing PDB files are needed,
 settings the simulations should be run at, analysis of the simulation results, etc.

You must follow the following criteria:
1) Return your plan as a Python list of subtasks that can be completed
in order to complete the specified task.
2) Ensure the response can be parsed by Python `json.loads`, e.g.:
no trailing commas,no single quotes, etc.
3) briefly explain your rationale why you choose this curriculum of subtasks
4) if we need to obtain any new tool or get any new files, include that
as a separate subtask in the relevant order
5) Only if you're asked to explore or find novel things: you should be able to
offer creative and interesting subtasks. You should be looking for
opportunities to discover as many diverse things as possible, accomplish as many
diverse tasks as possible to be expert at running molecular dynamics.
6) If you're asked to refine because the task failed, you should be able to offer
subtasks that can help the user complete the task.

You should only respond in JSON format as described below:
{{
    "Rationale": "rationale",
    "Plan": ["subtask1", "subtask2", "subtask3", ...]

}}

Here's the input:
- the final task: {final_task}
- a list of tools we currently have, if given: {tools}
- a list of files, if given: {files}
- failed subtasks, if given: {failed_tasks}
"""
CURRICULUM_INPUT_VARIABLES = [
    "final_task",
    "tools",
    "files",
    "failed_tasks",
]

curriculum_template = PromptTemplate(
    input_variables=CURRICULUM_INPUT_VARIABLES, template=CURRICULUM_PROMPT
)
