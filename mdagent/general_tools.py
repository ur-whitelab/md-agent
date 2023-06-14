from typing import Union

import pqapi
from langchain.tools import BaseTool


def litsearch(question: str) -> str:
    response = pqapi.agent_query("default", question)
    return response.answer


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = """Input a specific question,
                returns an answer from literature search."""
    pqa_key: str = ""
    openai_key: Union[str, None] = ""
    semantic_key: Union[str, None] = ""

    def __init__(
        self,
        pqa_key: str,
        openai_key: Union[str, None] = None,
        semantic_key: Union[str, None] = None,
    ):
        super().__init__()
        self.pqa_key = pqa_key
        self.openai_key = openai_key
        self.semantic_key = semantic_key

    def _run(self, question: str) -> str:
        """Use the tool"""
        return litsearch(question)

    async def _arun(self, question: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError
