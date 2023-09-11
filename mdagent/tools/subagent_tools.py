from typing import Optional

from langchain.tools import BaseTool

from . import Iterator, PathRegistry
from mdagent.subagents import SubAgentSettings, Iterator


class GetNewTool(BaseTool):
    name = "GetNewTool"
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
        Follow this format for your input:
        Tool: [tool description, input and output should be 1 string each]
        Context: [the full user prompt from the beginning, 1 string]  
    """

    # change line 23
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]

    def __init__(
        self, 
        path_registry: Optional[PathRegistry], 
        subagent_settings: Optional[SubAgentSettings]
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, query: str) -> str:
        """use the tool."""

        # check formatting
        try:
            lower_input = query.lower()
            task = lower_input.split("tool:", 1)[-1].split("context:", 1)[0].strip() ####
            context = lower_input.split("context:", 1)[-1].strip() ####
            if any(item in (None, "") for item in (task, context)):
                raise ValueError("incorrect input")
        except Exception:
            return "Incorrect input format. Please try again."

        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            if self.subagent_settings is None:
                return "Settings for subagents yet to be defined"
            
            # need to get the original_prompt; can MRKL provide it? 
            # run iterator
            iterator = Iterator(self.path_registry)
            tool_name = iterator.run(task, original_prompt) # update naming
            if tool_name:
                return f"""Tool created successfully: {tool_name}
                You can now use the tool in subsequent steps."""
            else:
                return "Failed to build a new tool."
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")




# below are other subagent-based tools (to be completed)

def add_new_skill(skillagent, code):
    #  similar to add_new_tool fxn from newtoolcreation.py
    # only difference is this looks at tools used during the entire ReAct's CoT
    # and create a consolidated tool & update skill library
    return ""

class SkillUpdate(BaseTool):
    name = "SkillUpdate"
    description = """

    ADD DESCRIPTION HERE

    """
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]

    def __init__(
        self, 
        path_registry: 
        Optional[PathRegistry], subagent_settings: Optional[SubAgentSettings]
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, code: str) -> str:
        """use the tool"""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            if self.agent is None:
                return "Agent for this tool not initialized"
            skill_result = add_new_skill(self.agent, code)
            return skill_result
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("This tool does not support async")


def code_retrieval():
    return ""

class SkillQuery(BaseTool):
    name = "SkillQuery"
    description = """

    ADD DESCRIPTION HERE

    """
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]

    def __init__(
        self, 
        path_registry: Optional[PathRegistry], 
        subagent_settings: Optional[SubAgentSettings]
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def _run(self, query: str) -> str:
        """use the tool"""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            if self.agent is None:
                return "Agent for this tool not initialized"
            query_result = code_retrieval(self.agent, query)
            return query_result
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("This tool does not support async")
