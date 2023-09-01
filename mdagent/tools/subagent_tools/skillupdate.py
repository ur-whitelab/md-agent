import os
import re
import json
import pickle
from typing import Optional
from langchain.tools import BaseTool

from mdagent.subagents.agents import Skill
from ..base_tools import PathRegistry


def add_new_skill(skillagent,code):
    #  similar to add_new_tool fxn from newtoolcreation.py
    # only difference is this looks at tools used during the entire ReAct's CoT
    # and create a consolidated tool & update skill library
    return ""


class SkillUpdate(BaseTool):
    name = "SkillUpdate"
    description = """

    ADD DESCRIPTION HERE

    """
    skillagent: Optional[Skill]
    path_registry: Optional[PathRegistry]

    def __init__(
        self, skillagent: Optional[Skill], path_registry: Optional[PathRegistry]
    ) -> str:
        super().__init__()
        self.path_registry = path_registry
        self.agent = skillagent

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