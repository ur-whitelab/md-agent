from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.subagents import SubAgentInitializer, SubAgentSettings
from mdagent.utils import PathRegistry


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
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]
    args_schema: Optional[Type[BaseModel]] = ExecuteSkillInputSchema

    def __init__(
        self,
        path_registry: Optional[PathRegistry] = None,
        subagent_settings: Optional[SubAgentSettings] = None,
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, skill_name, args=None):
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            if self.subagent_settings is None:
                return "Settings for subagents yet to be defined"
            agent_initializer = SubAgentInitializer(self.subagent_settings)
            skill_agent = agent_initializer.create_skill_manager(resume=True)
            if skill_agent is None:
                return "SubAgent for this tool not initialized"
            if args is not None:
                print("args: ", args)
                code_result = skill_agent.execute_skill_function(
                    skill_name, self.path_registry, **args
                )
            else:
                code_result = skill_agent.execute_skill_function(
                    skill_name, self.path_registry
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
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]
    args_schema: Optional[Type[BaseModel]] = SkillRetrievalInput

    def __init__(
        self,
        path_registry: Optional[PathRegistry] = None,
        subagent_settings: Optional[SubAgentSettings] = None,
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, query: str) -> str:
        """use the tool"""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            if self.subagent_settings is None:
                return "Settings for subagents yet to be defined"
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
