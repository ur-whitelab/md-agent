from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.subagents import SubAgentInitializer, SubAgentSettings


class ExecuteSkillInputSchema(BaseModel):
    """Input for Execute Skill"""

    skill_name: str = Field(description="Name of the skill to execute")
    args: Optional[dict] = Field(
        description="Input variables as a dictionary to pass to the skill"
    )


class ExecuteSkill(BaseTool):
    name = "ExecuteSkill"
    description = """Executes the code for a new tool or skill that has
    been recently made during the current iteration. Make sure to include
    function name and inputs arguments.
    """
    subagent_settings: Optional[SubAgentSettings]
    args_schema: Optional[Type[BaseModel]] = ExecuteSkillInputSchema

    def __init__(self, subagent_settings: Optional[SubAgentSettings] = None):
        super().__init__()
        self.subagent_settings = subagent_settings

    def _run(self, skill_name, args=None):
        try:
            path_registry = self.subagent_settings.path_registry
            agent_initializer = SubAgentInitializer(self.subagent_settings)
            skill_agent = agent_initializer.create_skill_manager(resume=True)
            if skill_agent is None:
                return "SubAgent for this tool not initialized"
            if args is not None:
                print("args: ", args)
                code_result = skill_agent.execute_skill_function(
                    skill_name, path_registry, **args
                )
            else:
                code_result = skill_agent.execute_skill_function(
                    skill_name, path_registry
                )
            return code_result
        except TypeError as e:
            return f"""{type(e).__name__}: {e}. Please check your inputs
            and make sure to use a dictionary.\n"""
        except ValueError as e:
            return f"{type(e).__name__}: {e}. Provide correct arguments of the skill.\n"
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e} \n"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("This tool does not support async")


class SkillRetrievalInput(BaseModel):
    """Input for Skill Retrieval"""

    query: str = Field(description="Query or task to retrieve skills as tools for")


class SkillRetrieval(BaseTool):
    name = "SkillRetrieval"
    description = """Only use this tool to retrieve skills that have been
    made during the current iteration. Use this tool less than other tools.
    """
    subagent_settings: Optional[SubAgentSettings]
    args_schema: Optional[Type[BaseModel]] = SkillRetrievalInput

    def __init__(self, subagent_settings: Optional[SubAgentSettings] = None):
        super().__init__()
        self.subagent_settings = subagent_settings

    def _run(self, query: str) -> str:
        """use the tool"""
        try:
            agent_initializer = SubAgentInitializer(self.subagent_settings)
            skill_agent = agent_initializer.create_skill_manager(resume=True)
            if skill_agent is None:
                return "SubAgent for this tool not initialized"
            skills = skill_agent.retrieve_skills(query)
            if skills is None:
                return "No skills found for this query"
            return f"\nFound {len(skills)} skills.\033[0m\n{list(skills.keys())}"
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("This tool does not support async")


class WorkflowPlanInputSchema(BaseModel):
    task: str = Field(
        description="""The final task you want to complete, ideally full
        user prompt you got from the beginning."""
    )
    curr_tools: str = Field(
        description="""List of all tools you have access to. Such as
        this tool, 'ExecuteSkill', 'SkillRetrieval', and maybe `Name2PDBTool`, etc."""
    )
    files: str = Field(description="List of all files you have access to.")
    # ^ would be nice if MDAgent could give files in case user provides unmapped files
    #   user-provided files should be mapped though
    failed_tasks: Optional[str] = Field(description="List of all failed tasks.")


class WorkflowPlan(BaseTool):
    name: str = "WorkflowPlan"
    description: str = """
        Useful at the beginning of solving a task, especially when it
        requires running simulations. It gives you a workflow plan for
        any Molecular Dynamics task or to explore.
        Also useful if you're stuck and need to refine your workflow plan.
    """
    args_schema: Type[BaseModel] = WorkflowPlanInputSchema
    subagent_settings: Optional[SubAgentSettings]

    def __init__(self, subagent_settings: Optional[SubAgentSettings] = None):
        super().__init__()
        self.subagent_settings = subagent_settings

    def _run(self, task, curr_tools, files, failed_tasks=""):
        try:
            agent_initializer = SubAgentInitializer(self.subagent_settings)
            curriculum_agent = agent_initializer.create_curriculum_agent()
            if curriculum_agent is None:
                return "Curriculum Agent is not initialized"
            if files == "":
                files = self.path_registry.list_path_names()
            rationale, decomposed_tasks = curriculum_agent.run(
                task, curr_tools, files, failed_tasks
            )
            return f"""Here's the list of subtasks decomposed from the main task:\n
                {decomposed_tasks}. \n Now, do these subtasks one by one."""
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
