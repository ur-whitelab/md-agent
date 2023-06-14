import os

import pytest

from mdagent.general_tools import Scholar2ResultLLM


@pytest.fixture
def question():
    qs = "What are the effects of norhalichondrin B in mammals?"
    return qs


@pytest.fixture
def litsearch():
    return Scholar2ResultLLM(os.getenv("PQA_API_KEY"))


def test_litsearch(question, litsearch):
    os.getenv("PQA_API_KEY")
    os.getenv("OPENAI_API_KEY")
    os.getenv("SEMANTIC_API_KEY")
    pqa_result = litsearch(question)
    result = pqa_result._run(question)
    assert isinstance(result, str)
    assert len(result) > 0
