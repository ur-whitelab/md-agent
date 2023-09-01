action_format = """
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
"""

action_inputs = """
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
"""

action_prefix = """
You are a helpful assistant that writes python code to complete any
OpenMM or other molecular dynamics related task specified by me.

I will give you the following: {action_inputs}

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

You should only respond in the format as described below:

RESPONSE FORMAT:
{action_format}
"""

action_prompt = """
INPUT:
recent_history: {recent_history},
full_history: {full_history},
skills: {skills}
"""

# if first iteration:

action_prefix_1 = """
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
Plan: How to complete the task step by step.
    You should pay attention to files since it tells what you files you have access to.
    The task completeness check partially depends on your final files list.
Code:
    1) Write a function taking a string as the only argument.
    3) Your function will be reused for building more complex functions.
        Therefore, you should make it generic and reusable.
        You should always check whether you have the required files before using them.
        If not, you should first collect the required files
        and reuse the above useful programs.
    4) Functions in the iteration history will not be saved or executed.
        Do not reuse functions listed there.
    5) Anything defined outside a function will be ignored,
        define all your variables inside your functions.
    6) Your function input and output MUST be a string.
        If you need to pass in an object, you should convert it to a string first.
        If you need to pass in a file, you should pass in the
        path to the file as a string.
        If you need to output a file, you should instead save the file and
        return the path to the file as a string.
    7) Do not write infinite loops or recursive functions.
    8) Name your function in a meaningful way (can infer the task from the name).

You should only respond in the format as described below:

RESPONSE FORMAT:
{action_format}
"""

action_prompt_1 = """
INPUT:
files: {files},
task: {task},
context: {context},
skills: {skills},
"""

