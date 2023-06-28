import os

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel

from ..tools import (
    AddHydrogensCleaningTool,
    CheckDirectoryFiles,
    #InstructionSummary,
    ListRegistryPaths,
    MapPath2Name,
    Name2PDBTool,
    PackMolTool,
    PlanBVisualizationTool,
    RemoveWaterCleaningTool,
    Scholar2ResultLLM,
    SerpGitTool,
    SetUpAndRunTool,
    SimulationOutputFigures,
    SpecializedCleanTool,
    VisualizationToolRender,
    # AvgRmsdTrajectoryTool,
    # PPIDistanceTool,
    # RmsdCompareTool,
    # RmsdTrajectoryTool,
)
from ..utils import PathRegistry


def make_tools(llm: BaseLanguageModel, verbose=False):
    load_dotenv()

    all_tools = agents.load_tools(["python_repl", "human", "llm-math"], llm)

    # add tools

    all_tools += [
        VisualizationToolRender(),
        CheckDirectoryFiles(),
        PlanBVisualizationTool(),
        SpecializedCleanTool(),
        RemoveWaterCleaningTool(),
        AddHydrogensCleaningTool(),
        SetUpAndRunTool(),
        Name2PDBTool(),
        SimulationOutputFigures(),
        #InstructionSummary(),
        # PPIDistanceTool(),
        # RmsdCompareTool(),
        # RmsdTrajectoryTool(),
        # AvgRmsdTrajectoryTool(),
    ]

    # add registry tools
    # get instance first
    path_instance = PathRegistry.get_instance()
    # add tools
    all_tools += [
        SetUpAndRunTool(path_registry=path_instance),
        ListRegistryPaths(path_registry=path_instance),
        MapPath2Name(path_registry=path_instance),
        PlanBVisualizationTool(path_registry=path_instance),
        Name2PDBTool(path_registry=path_instance),
        SpecializedCleanTool(path_registry=path_instance),
        RemoveWaterCleaningTool(path_registry=path_instance),
        AddHydrogensCleaningTool(path_registry=path_instance),
        PackMolTool(path_registry=path_instance),
    ]
    
    # Get the api keys
    serp_key = os.getenv("SERP_API_KEY")
    if serp_key:
        all_tools.append(SerpGitTool(serp_key))  # add serpapi tool
    pqa_key = os.getenv("PQA_API_KEY")
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key)) # add literature search tool
    return all_tools
