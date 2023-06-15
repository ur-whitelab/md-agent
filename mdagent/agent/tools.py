import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from mdagent.general_tools import Scholar2ResultLLM


def make_tools(llm: BaseLanguageModel, verbose=False):
    os.getenv("OPENAI_API_KEY")
    pqa_key = os.getenv("PQA_API_KEY")

    all_tools = agents.load_tools(["python_repl", "human", "llm-math"], llm)

    all_tools += []
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key))
    return all_tools