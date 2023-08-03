import pqapi
from langchain.tools import BaseTool


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = """Input a specific question,
                returns an answer from literature search."""

    pqa_key: str = ""

    def __init__(self, pqa_key: str):
        super().__init__()
        self.pqa_key = pqa_key

    def _run(self, question: str) -> str:
        """Use the tool"""
        try:
            response = pqapi.agent_query("default", question)
            return response.answer
        except Exception:
            return "Literature search failed."

    async def _arun(self, question: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError
