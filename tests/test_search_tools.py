import os

import pytest
from langchain.chat_models import ChatOpenAI

from mdagent.tools.base_tools import Scholar2ResultLLM


@pytest.fixture
def questions():
    qs = [
        "What are the effects of norhalichondrin B in mammals?",
    ]
    return qs[0]


@pytest.mark.skip(reason="This requires an API call")
def test_litsearch(questions, get_registry):
    llm = ChatOpenAI()

    searchtool = Scholar2ResultLLM(llm=llm, path_registry=get_registry("raw", False))
    for q in questions:
        ans = searchtool._run(q)
        assert isinstance(ans, str)
        assert len(ans) > 0
    if os.path.exists("../query"):
        os.rmdir("../query")