import os

import pytest

from mdagent.general_tools import Scholar2ResultLLM


@pytest.fixture
def question():
    qs = "What are the effects of norhalichondrin B in mammals?"
    return qs


class LitSearchTester:
    def __init__(self, pqa_key, openai_api_key, semantic_api_key):
        self.pqa_key = pqa_key
        self.openai_api_key = openai_api_key
        self.semantic_api_key = semantic_api_key

    def run(self, question):
        pqa_result = Scholar2ResultLLM(self.pqa_key)
        result = pqa_result._run(question)
        return result


def test_litsearch(question):
    pqa_key = os.getenv("PQA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    semantic_api_key = os.getenv("SEMANTIC_API_KEY")
    pqa_result = LitSearchTester(pqa_key, openai_api_key, semantic_api_key)
    result = pqa_result.run(question)
    assert isinstance(result, str)
    assert len(result) > 0
