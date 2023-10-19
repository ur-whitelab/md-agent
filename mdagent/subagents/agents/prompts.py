from langchain.prompts import PromptTemplate

action_template = PromptTemplate(
    inputs=["files", "task", "history", "skills"],
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
        You should pay attention to files since it tells what you files you have created.
        The task completeness check partially depends on your final files list.
    Code:
        1) Write a function taking a string as the only argument.
        3) Your function will be reused for building more complex functions.
            Therefore, you should make it generic and reusable.
            You should always check whether you have the required files before using them.
            If not, you should first collect the required files and reuse
            the given pre-existing skills.
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
        9) Include all imports necessary for your code to run. If possible, include these
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
    files: {files}
    task: {task}
    history: {history}
    skills: {skills}
        """,
)

critic_template=PromptTemplate(
    inputs=["code", "code_output", "task"],
    template="""
    You are an assistant that assesses the quality of my code
    and the validity of its output for my molecular dynamics project,
    and provides useful guidance.

    I will give you the following information:
    
    Code: The source code
    Code Output: The output of the executed code
    Task: The objective that the code needs to accomplish
    
    Your job is to evaluate the following:
    task relevance: whether the code meets the task requirements (note,
        this is not the same as whether the code is syntactically correct, but rather
        whether it is written to accomplish the task)
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
    task: {task}
    """
)


skill_describe_template = PromptTemplate(
    inputs=["code"],
    template="""
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

    You should only respond in the format as described below:

    RESPONSE FORMAT:
    Function name: this should be Python function name
    Tool name: this should be Python name for class.
    Tool description: this should be tool description in format of docstrings.

    Here is the code:
    {code}
    """,
)

skill_wrapper_template = PromptTemplate(
    inputs=["code", "fxn_name", "tool_name", "description"],
    template="""
    You are a helpful assistant that writes a full code, the contents in a
    complete python script for the given basic code & other essential pieces of
    information.

    I will give you the python code, function name, tool name, and description.

    You must follow the following criteria:
    1) the given code must be wrapped in a Python function, naming it with
    the given function name.
    2) below the function code, create a class as BaseTool object, naming it
    with the given tool name. Basetool object must has the following exact
    functions: __init__, _run, and _async.
    3) the full code must contain all imports.
    4) the full code must contain the full description of the tool in docstrings.
    Do not modify or shorten the given description.
    5) PathRegistry must be added as an argument in the class initialization.
    6) the full code must follow closely to the following example format
    as much as possible, <FIELDS> tells you where you can place the given
    inputs.
    7) Don't add anything after BaseTool class definition. Async _async function
    must be the last part in the file.

    Example format of the full code:
    Full Code: ```from typing import Optional

    <ANY OTHER IMPORTS HERE>
    from langchain.tools import BaseTool

    from mdagent.utils import PathRegistry


    def <FXN_NAME>(<FXN_INPUTS>, PathRegistry):
        <CODE HERE>

    class <TOOL_NAME>(BaseTool):
        name = <TOOL_NAME>
        description = <INSERT FULL GIVEN DESCRIPTION HERE>
        path_registry: Optional[PathRegistry]

        def __init__(self, path_registry: Optional[PathRegistry]):
            super().__init__()
            self.path_registry = path_registry

        def _run(self, <FXN_INPUTS>: str) -> str:
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
            raise NotImplementedError("This tool does not support async")
    ```
    You must only respond with the full code in the format as described above.

    Here is the input:
    code: {code},
    fxn_name: {fxn_name},
    tool_name: {tool_name},
    description: {description}
    """,
)
