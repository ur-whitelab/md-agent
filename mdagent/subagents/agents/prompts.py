from langchain.prompts import PromptTemplate

action_template = PromptTemplate(
    input_variables=["files", "task", "history", "skills"],
    template="""
    You are a helpful assistant that writes python code to complete any
    OpenMM or other molecular dynamics related task specified by me.
    I will give you the following:
    1. The files you may access in your code
    2. The task you must complete
    3. The previous iterations in the conversation. You should learn from these.
    4. The skills you have learned so far. You may reuse them in your code or use them
        to help you write your code if needed.
    You should then respond to me with
    Explain (if applicable):
        1. Are there any steps missing in your plan?
        2. Why does the code not complete the task?
        3. What does the code and execution error imply?
    Plan: How to complete the task step by step.
        If there was an execution error, you should try to solve it.
        You should pay attention to files since it tells
        what you files you have access to.
        The task completeness check partially depends on your final files list.
    Code:
        1) Write a function taking a string as the only argument.
        3) Your function will be reused for building more complex functions.
            Therefore, you should make it generic and reusable.
            You should always check whether you have
            the required files before using them.
        4) Functions in the given history summary
            section will not be saved or executed.
            Do not reuse functions listed there.
        5) Anything defined outside a function will be ignored,
            define all your variables inside your functions.
        6) Your function input and output MUST be a string.
            If you need to pass in an object, you should convert it to a string first.
            If you need to pass in a file, you should pass in the
            path to the file as a string.
            If you need to output a file, you should instead save the file
            and return the path to the file as a string.
        7) Do not write infinite loops or recursive functions.
        8) Name your function in a meaningful way (can infer the task from the name).
        9) Include all imports necessary for your code to run.
            If possible, include these
            imports in the function itself.
        10) At the end of your code, call the function you defined with the input.
        11) Don't use ... in any of your code. It should be complete and ready
            for execution.
    You should only respond the following format:
    Explain: ...
    Plan:
    1) ...
    2) ...
    3) ...
    ...
    Code:
    ```
    # helper functions (only if needed, try to avoid them)
    ...
    # main function after the helper functions
    def yourMainFunctionName(query):
    # ...
    ```
    Here is the input:
    files: {files},
    task: {task},
    history: {history},
    skills: {skills},
        """,
)

critic_template = PromptTemplate(
    input_variables=["code", "code_output", "task"],
    template="""
    You are an assistant that assesses the quality of my code
    and the validity of its output for my molecular dynamics project,
    and provides useful guidance.
    I will give you the following information:
    Code: The source code
    Code Output: The output of the executed code
    Task: The objective that the code needs to accomplish
    Your job is to evaluate the following:
    task relevance: whether the code is written to meet the task requirements (note,
        this is not the same as whether the code is syntactically correct, but rather
        whether it is written to accomplish the task.
    critique: you should always provide a critique of the code, even if it is successful
    suggestions: you should provide suggestions for how to improve the code,
        even if it is successful
    You should only respond in JSON format as described below:
    {{
        "task_relevance": boolean,
        "critique": "critique",
        "suggestions": "suggestions"
    }}
    Here is the input:
    code: {code},
    code_output: {code_output},
    task: {task},

    IMPORTANT: the code author is a student, so you
    should not be too strict when assigning the task
    relevance or providing critique.
    """,
)

skill_template = PromptTemplate(
    input_variables=["code"],
    template="""
    You are a helpful assistant that writes a description of the given python code.

    I will give you the python function code.

    You must follow the following criteria:
    1) Try to summarize the function description in no more than 6 sentences.
    2) The description should contain expected inputs and outputs and explain
    clearly what the code does.
    3) Description should be in docstrings with up to 80 characters per line.
    You should only respond with docstrings:

    this should be tool description in format of docstrings
    Here is the code:
        {code}
        """,
)

curriculum_template = PromptTemplate(
    input_variables=["final_task", "tools", "files", "failed_tasks"],
    template="""
    You are an expert molecular dynamics scientist and your role is to set
    a curriculum plan with subtasks to complete the final task specified by me.
    This is similar to molecular dynamics workflow planning that includes
    determining what kind of preprocessing PDB files are needed,
    settings the simulations should be run at, analysis of the simulation results, etc.


    You must follow the following criteria:
    1) Return your plan as a Python list of subtasks that can be completed
    in order to complete the specified task.
    2) Ensure the response can be parsed by Python `json.loads`, e.g.:
    no trailing commas, no single quotes, etc. Don't start with ```json.
    3) briefly explain your rationale why you choose this curriculum of subtasks
    4) For each subtask, specify which tool you should use. If and only if
    there is no suitable tool, mention that we need to obtain a new tool.
    5) Each subtask should be very specific and essential to the completion
    of the task. In other words, fewer steps are preferred over more wasteful steps.
    Don't suggest preprocessing PDB files or running simulations unless it's
    absolutely necessary or requested so.
    6) REFINE: If you're asked to make a plan because some task failed, you
    should be able to refine and help complete the task.
    7) EXPLORE: If and only if you're asked to explore or find novel things:
    you should be able to offer creative and interesting subtasks. You should
    be looking for opportunities to discover as many diverse things as possible,
    accomplish as many diverse tasks as possible to be expert at running
    molecular dynamics.

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
    """,
)
