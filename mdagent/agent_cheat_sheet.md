# Organization Plan
```
├── .github
├── mdagent
│   ├── _init_.py
│   ├── mainagent
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── mdagent_prompt.py
│   ├── subagents
│   │   ├── __init__.py
│   │   ├── subagent_fxns.py    # contains multiagent functions
│   │   ├── subagent_setup.py   # contains SubAgentInitializer
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── skill.py
│   │   │   ├── currciulum.py
│   │   │   ├── action.py
│   │   │   ├── task_critic.py
│   │   │   ├── code_critic.py
│   │   ├── prompts
│   │   │   ├── __init__.py
│   │   │   ├── action_prompts.py
│   │   │   ├── critic_prompts.py
│   │   │   ├── curriculum_prompts.py
│   │   │   ├── skill_prompts.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── maketools.py
│   │   ├── subagent_tools.py
│   │   ├── base_tools
│   │   │   └── ...
└── notebooks
│   │   ├── ...
└── tests
│   │   ├── ...
├── ...
```

# Import Chain
Top-level to lower-level dependencies to prevent circular imports
```
mainagent   - depends on everything below
  ↑
tools       - depends on subagents, utils
  ↑
subagents   - depends on utils
  ↑
utils       - depends on nothing
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

## ActionAgent - 1st
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

## ActionAgent - resume
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

## Code Critic Agent
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

## Task Critic Agent
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

## Skill Manager
- creates tool_name and description for new code, wraps it into Langchain tool
- also stores everything in 'skill_library' directory
- input:
    - fxn_name
    - code
- output:
    - langchain tool name
- Lives whenever new code is successful and needs to store as a tool
- SkillManager.add_new_tool(fxn_name, code)

## Curriculum Agent
- proposes a curriculum of subtasks to achieve the original prompt. Useful
at the beginning of MDAgent's cycle or whenever MDAgent gets stuck. Also
useful when the user wants to explore.
- inputs:
    - original_task (user prompt)
    - current list of tools/skills (from MDAgent and skill library)
    - files
    - failed tasks, if any
- output:
    - the rationale for the plan of subtasks
    - a list of subtasks
- Lives outside iterator code, as a separate tool.
- CurriculumAgent.run(original_task, tools, files, failed_tasks)
