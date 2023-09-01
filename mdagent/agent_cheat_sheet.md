# Organization Plan
```
├── .github
├── mdagent
│   ├── _init_.py
│   ├── mainagent
|   |   ├── __init__.py
│   │   ├── agent.py
|   |   ├── mdagent_prompt.py
│   ├── subagents
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── skill.py
│   │   │   ├── currciulum.py
│   │   │   ├── action
│   │   │   ├── task_critic
│   │   │   ├── code_critic
│   │   ├── prompts
│   │   │   ├── __init__.py
│   │   │   ├── action_prompts.py
│   │   │   ├── critic_prompts.py
│   │   │   ├── curriculum_prompts.py
│   │   │   ├── skill_prompts.py
│   ├── tools
|   |   ├── __init__.py
│   │   ├── base_tools
|   |   ├── subagent_tools
│   │   |   ├── iteration.py
└── notebooks
│   │   ├── ...
└── tests
│   │   ├── ...
├── ...
```

# Agent Cheat Sheet - for development

We are working with multiple agents, who interact with each other in various ways. To make this process more seamless, here is an agent cheat sheet.

For each agent, please use the following guide:

AGENT NAME
-purpose
-input(s)
-output(s)
-place within agent framework
-how to call

For example, curriculum's "place within agent framework" might be first step in iteration, after 1st iteration and as a tool within mrkl. This is a rather complex example, but you get the point.

## Action - 1st
- runs code, executes code
- inputs:
    - files
    - task
    - context (user prompt)
    - skills
- outputs:
    - success (boolean, did it execute)
    - proposed code
    - code output (or error)
- first step in first iteration only
- Action._run_code

## Action - resume
- runs code, executes code
- inputs:
    - recent history
    - full_history
    - skills
- outputs:
    - success (boolean, did it execute)
    - proposed code
    - code output (or error)
- lives after curriculum, before code critic in iteration (then action <> critics)
- Action._run_code

## Code Critic
- critique code & determine if pass/fail and how to improve
- inputs:
    - code (from action)
    - code output (from action)
    - task (from curriculum or MRKL)
    - context (user prompt)
- outputs:
    - critique includes:
        - code quality, validity, success (boolean), critique, suggestions
- Lives after action in iteration
    - if success = True --> call task critic
    - if success = False --> back to action or curriculum (if max iter)
- CodeCritic._run

## Task Critic
- critique whether valid code addresses prompt
- inputs:
    - files
    - code (from action)
    - code output (from action)
    - task (from curriculum or MRKL)
    - context (user prompt)
    - additional information (optional)
        - used only if invalid formatting on 1st attempt
- outputs:
    - success (boolean)
    - task critique (if success = False)
- Lives after code critic & is only envoked if code critic deems code successful.
    - if successful task critic, add to skill library
    - if unsuccessful, continue
- TaskCritic._run_task_critic

## Skill Agent (for creating a new tool)
- creates tool_name and description for new code, wraps it into Python function and Langchain tool
- input:
    - code
- output:
    - langchain tool name (in case we can use right away. If not, it doesn't need to pass anything for the current ReAct prompt?)
    - (created .py file with python function & langchain tool)
- place within agent framework

## Refining Curriculum Agent (to refine task if code keeps failing)
- proposes a new, refined task closely aligned to the 'spirit' of user prompt as much as possible
- inputs:
    - original_task (user prompt)
    - recent history
    - full history (key info: failed/completed tasks)
    - current list of tools/skills (from MRKL and skill library)
    - files
- output:
    - a new task
- Lives after action keeps failing to create successful code
- RefiningCurriculumAgent._run


## Outside 'Iteration' tool (for exploring, storing skill library)

### Skill Agent (for creating a new SKILL)
- creates tool_name and description for a 'new' user input and a collection of tools, store it as a skill (and wrap it into Langchain tool)
- inputs:
    - original
- output(s)
-

### SkillQuery tool
- just call skill agent to query tools/skill library

### 'Explorer' Curriculum Agent
- proposes a new prompt for MKRL agent
- inputs:
    - original_task (user prompt)
    - recent history (latest task)
    - full history (key info: failed/completed tasks)
    - current list of tools/skills (from MRKL and skill library)
    - files
- output:
    - a new task
