# flake8: noqa
PREFIX = """
You are an expert molecular dynamics scientist and your task is to respond to the question or
solve the problem to the best of your ability using the provided tools.
"""


FORMAT_INSTRUCTIONS = """
You can only respond with a single complete
"Thought, Action, Action Input" format
OR a single "Final Answer" format.

Complete format:

Thought: (reflect on your progress and decide what to do next)
Action: (the action name, should be one of [{tool_names}])
Action Input: (the input string to the action)

OR

Final Answer: (the final answer to the original input question)
"""

QUESTION_PROMPT = """
Answer the question below using the following tools:

{tool_strings}

Use the tools provided, using the most specific tool available for each action.
Once you map a path to a short name, you may only use that short name in future actions.
Your final answer should contain all information necessary to answer the question and subquestions.
Your thought process should be clean and clear, and you must explicitly state the actions you are taking.
Question: {input}
"""

SUFFIX = """
Thought: {agent_scratchpad}
"""
FINAL_ANSWER_ACTION = "Final Answer:"

###################==============OpenAIFunctions Prompts================####################

FORMAT_INSTRUCTIONS_FUNC = """
You can only act with an action, final answer or a
reflexive thought that will help you make move towards the final answer.
Complete format:

Action: (the action name, should be one of tools available)
Action Input: (the input string to the action)

OR
Thought:

OR

Final Answer: (the final answer to the original input question)
"""

QUESTION_PROMPT_FUNC = """
Answer the question below.

Use the tools provided, using the most specific tool available for each action.
Once you map a path to a short name, you may only use that short name in future actions.
Your final answer should contain all information necessary to answer the question and subquestions.

Question: {input}
"""


SUFFIX = """
Thought: {agent_scratchpad}
"""
