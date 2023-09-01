import os
import re
import json
import pickle
from typing import Optional
from langchain.tools import BaseTool

from mdagent.subagents.agents import Skill
from ..base_tools import PathRegistry


def code_retrieval():
    return ""


class SkillQuery(BaseTool):
    name = "SkillQuery"
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