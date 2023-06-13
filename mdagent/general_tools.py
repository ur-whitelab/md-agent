import pqapi
from langchain.tools import BaseTool


def dummy_function() -> int:
    return 46


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
        response = pqapi.agent_query("default", question)
        return response.answer

    async def _arun(self, question: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError
