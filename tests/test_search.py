import os

import pytest

from mdagent.general_tools import Scholar2ResultLLM


@pytest.fixture
def question():
    qs = "What are the effects of norhalichondrin B in mammals?"
    return qs


@pytest.fixture
def litsearch():
    pqa = os.getenv("PQA_API_KEY")
    openai = os.getenv("OPENAI_API_KEY")
    semantic = os.getenv("SEMANTIC_API_KEY")
    return Scholar2ResultLLM(pqa, openai, semantic)


def test_litsearch(question, litsearch):
    result = litsearch._run(question)
    assert isinstance(result, str)
    assert len(result) > 0
