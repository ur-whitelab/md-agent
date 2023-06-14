import os

import pytest

from mdagent.general_tools import Scholar2ResultLLM


@pytest.fixture
def question():
    qs = "What are the effects of norhalichondrin B in mammals?"
    return qs


# @pytest.mark.skip()
def test_litsearch(question):
    pqa_key = os.getenv("PQA_API_KEY")
    pqa_result = Scholar2ResultLLM(pqa_key)
    result = pqa_result._run(question)

    assert isinstance(result, str)
    assert len(result) > 0
