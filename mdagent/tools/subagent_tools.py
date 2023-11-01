from typing import List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from mdagent.subagents import Iterator, SubAgentInitializer, SubAgentSettings
from mdagent.utils import PathRegistry

from .maketools import get_all_tools_string


class CreateNewToolInput(BaseModel):
    """Input for Create New Tool"""

    task: str = Field(..., description="Description of task the tool should perform")
    orig_prompt: str = Field(..., description="Full user prompt from the beginning")
    curr_tools: List[str] = Field(..., description="List of tools the agent has")


class CreateNewTool(BaseTool):
    name = "CreateNewTool"
    description = """
        This tool is used to create a new tool.
        Given a description of the tool needed,
        it will write and test tools.
        If this tool hits maximum iterations without suceeding
        to create a tool, it will return a failure. If you
        receive a failure, you can try again with a different
        input description. If you receive a success, you will
        recieve the tool name, description, and input type.
        You can then use the tool in subsequent steps.
    """
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]
    arg_schema: Optional[Type[BaseModel]] = CreateNewToolInput

    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        subagent_settings: Optional[SubAgentSettings],
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, task: str, orig_prompt: str, curr_tools: List[str]) -> str:
        """use the tool."""
        # check formatting
        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            if self.subagent_settings is None:
                return "Settings for subagents yet to be defined"
            # run iterator
            newcode_iterator = Iterator(self.path_registry, self.subagent_settings)
            all_tools = get_all_tools_string()
            tool_name = newcode_iterator.run(task, orig_prompt, curr_tools, all_tools)
            if tool_name:
                return f"""Tool created successfully: {tool_name}
                You can now use the tool in subsequent steps."""
            else:
                return "The 'CreateNewTool' tool failed to build a new tool."
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"

    async def _arun(self, task: str, orig_prompt: str, curr_tools: List[str]) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


class ExecuteSkillInput(BaseModel):
    """Input for Execute Skill"""

    tool_name: str = Field(..., description="Name of the skill or tool to execute")
    args: Optional[dict] = Field(
        ..., description="Input variables as a dictionary to pass to the skill"
    )


class ExecuteSkill(BaseTool):
    name = "ExecuteSkill"
    description = """Executes the code for a new tool or skill that has
    been recently made during the current iteration.
    """
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]
    arg_schema: Optional[Type[BaseModel]] = ExecuteSkillInput

    def __init__(
        self,
        path_registry: Optional[PathRegistry] = None,
        subagent_settings: Optional[SubAgentSettings] = None,
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, tool_name: str, args: Optional[dict] = None) -> str:
        """use the tool"""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            if self.subagent_settings is None:
                return "Settings for subagents yet to be defined"
            agent_initializer = SubAgentInitializer(self.subagent_settings)
            skill_agent = agent_initializer.create_skill_manager(resume=True)
            if skill_agent is None:
                return "Agent for this tool not initialized"
            if args is not None:
                code_result = skill_agent.execute_skill_code(
                    tool_name, skill_agent, self.path_registry, **args
                )
            else:
                code_result = skill_agent.execute_skill_code(
                    tool_name, skill_agent, self.path_registry
                )
            return code_result
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("This tool does not support async")


class SkillRetrievalInput(BaseModel):
    """Input for Skill Retrieval"""

    query: str = Field(..., description="Query or task to retrieve skills as tools for")


class SkillRetrieval(BaseTool):
    name = "SkillRetrieval"
    description = """Use this tool to retrieve a list of relevant skills.
    Useful to check current skills before creating a new skill/tool or
    to see if a skill exists."""
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
