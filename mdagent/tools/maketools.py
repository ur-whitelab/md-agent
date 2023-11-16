import os

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel

from mdagent.utils import PathRegistry

from .base_tools.clean_tools import (
    AddHydrogensCleaningTool,
    RemoveWaterCleaningTool,
    SpecializedCleanTool,
)
from .base_tools.git_issues_tool import SerpGitTool
from .base_tools.md_util_tools import Name2PDBTool
from .base_tools.pdb_tools import PackMolTool
from .base_tools.plot_tools import SimulationOutputFigures
from .base_tools.ppi_tools import PPIDistance
from .base_tools.registry_tools import ListRegistryPaths, MapPath2Name
from .base_tools.rmsd_tools import RMSDCalculator
from .base_tools.search_tools import Scholar2ResultLLM
from .base_tools.setup_and_run import SetUpAndRunTool
from .base_tools.vis_tools import (
    CheckDirectoryFiles,
    PlanBVisualizationTool,
    VisualizationToolRender,
)


def make_all_tools(llm: BaseLanguageModel, verbose=False):
    load_dotenv()

    all_tools = agents.load_tools(["python_repl", "human", "llm-math"], llm)

    # add tools
    all_tools += [
        VisualizationToolRender(),
        CheckDirectoryFiles(),
        SimulationOutputFigures(),
        PPIDistance(),
        RMSDCalculator(),
    ]

    # get instance first
    path_instance = PathRegistry.get_instance()
    # add tools
    all_tools += [
        AddHydrogensCleaningTool(path_registry=path_instance),
        # InstructionSummary(path_registry=path_instance),
        ListRegistryPaths(path_registry=path_instance),
        MapPath2Name(path_registry=path_instance),
        Name2PDBTool(path_registry=path_instance),
        PackMolTool(path_registry=path_instance),
        PlanBVisualizationTool(path_registry=path_instance),
        RemoveWaterCleaningTool(path_registry=path_instance),
        SetUpAndRunTool(path_registry=path_instance),
        SpecializedCleanTool(path_registry=path_instance),
    ]

    # Get the api keys
    serp_key = os.getenv("SERP_API_KEY")
    if serp_key:
        all_tools.append(SerpGitTool(serp_key))  # add serpapi tool
    pqa_key = os.getenv("PQA_API_KEY")
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key))  # add literature search tool
    return all_tools
