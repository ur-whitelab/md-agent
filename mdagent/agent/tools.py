import os

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel

from ..tools.clean_tools import (
    AddHydrogensCleaningTool,
    RemoveWaterCleaningTool,
    SpecializedCleanTool,
)
from ..tools.md_util_tools import Name2PDBTool
from ..tools.search_tools import InstructionSummary, Scholar2ResultLLM
from ..tools.setup_and_run import SetUpAndRunTool
from ..tools.vis_tools import (
    CheckDirectoryFiles,
    PlanBVisualizationTool,
    VisualizationToolRender,
)


def make_tools(llm: BaseLanguageModel, verbose=False):
    load_dotenv()

    # Get the api keys

    os.getenv("OPENAI_API_KEY")
    pqa_key = os.getenv("PQA_API_KEY")

    all_tools = agents.load_tools(["python_repl", "human", "llm-math"], llm)

    # add visualization tools

    all_tools += [
        CheckDirectoryFiles(),
        VisualizationToolRender(),
        InstructionSummary(),
        PlanBVisualizationTool(),
        SpecializedCleanTool(),
        RemoveWaterCleaningTool(),
        AddHydrogensCleaningTool(),
        SetUpAndRunTool(),
        Name2PDBTool(),
    ]

    # add literature search tool
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key))
    return all_tools
