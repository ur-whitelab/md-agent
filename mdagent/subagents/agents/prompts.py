from langchain.prompts import PromptTemplate

action_template_1 = PromptTemplate(
    inputs=["files", "task", "context", "skills"],
    template="""
    You are a helpful assistant that writes python code to complete any
    OpenMM or other molecular dynamics related task specified by me.

    I will give you a task and context.
    You should then write a python function that completes the task in the context.

    You will also have access to all skills you have learned so far.
    You may reuse them in your code or use them to help you write your code if needed.
    The skills will be a dictionary of name and function pairs.

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
    context: {context}
    skills: {skills}
        """
)

action_template = PromptTemplate(
    inputs=["recent_history", "full_history", "skills"],
    template="""
    You are a helpful assistant that writes python code to complete any
    OpenMM or other molecular dynamics related task specified by me.
    
    I will give you the following:
    Recent History:
        1. The most recent completed iteration
        2. The task you must complete
        3. The context of the task
        4. The code written in the last iteration
        5. The output of the code written in the last iteration
        6. All available files from the last iteration
        7. The code critique from the last iteration
        8. The task critique from the last iteration, if applicable

    I will also give you all data from the beginning of the conversation,
    the Full History
        1. Each Iteration Number
        2. The tasks
        3. The context of the tasks
        4. The code written in each iteration
        5. The output of the code written in the each iteration
        6. All available files from the each iteration
        7. The code critique from the each iteration
        8. The task critique from the each iteration, if applicable

    You will also have access to all Skills you have learned so far.
    You may reuse them in your code or use them to help you write your code if needed.
    The Skills will be a dictionary of name and function pairs.
    
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
    recent_history: {recent_history}
    full_history: {full_history}
    skills: {skills}
    """
)

code_critic_template = PromptTemplate(
    inputs=["code", "code_output", "task", "context"],
    template="""
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
    Code Output: The output of the executed code
    Task: The objective that the code needs to accomplish
    
    You should only respond in JSON format as described below:
    {{
        "code_quality": "code_quality",
        "output_validity": "output_validity",
        "success": boolean,
        "critique": "critique",
        "suggestions": "suggestions"
    }}
    Ensure the response can be parsed by Python `json.loads`,
    e.g.: no trailing commas, no single quotes, etc.
    
    Here is the input:
    code: {code}
    code_output: {code_output}
    task: {task}
    context: {context}
    """
)

task_critic_template = PromptTemplate(
    inputs=["files", "code", "code_output", "task", "context", "additional_information"],
    template="""
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

    You should only respond in JSON format as described below:
    {{
        "success": boolean,
        "critique": string,
    }}
    Ensure the response can be parsed by Python `json.loads`,
    e.g.: no trailing commas, no single quotes, etc.

    Here is the input:
    INPUT:
    files: {files},
    code: {code},
    output: {code_output},
    task: {task},
    context: {context},
    additional_information: {additional_information}
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
    """
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
    """
)