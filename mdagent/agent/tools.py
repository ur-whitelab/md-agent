import os

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel

from ..tools import (
    AddHydrogensCleaningTool,
    CheckDirectoryFiles,
    FullRegistry2File,
    ListRegistryObjects,
    ListRegistryPaths,
    MapPath2Name,
    Name2PDBTool,
    Objects2File,
    OpenMMObjectRegistry,
    PathRegistry,
    Paths2File,
    PlanBVisualizationTool,
    RemoveWaterCleaningTool,
    Scholar2ResultLLM,
    SetUpAndRunTool,
    SpecializedCleanTool,
    VisualizationToolRender,
)


def make_tools(llm: BaseLanguageModel, verbose=False):
    load_dotenv()

    all_tools = agents.load_tools(["python_repl", "human", "llm-math"], llm)

    # add visualization tools

    all_tools += [
        VisualizationToolRender(),
        CheckDirectoryFiles(),
    ]

    # add registry tools
    # get instance first
    path_instance = PathRegistry.get_instance()
    object_instance = OpenMMObjectRegistry.get_instance()
    # add tools
    all_tools += [
        FullRegistry2File(path_registry=path_instance, object_registry=object_instance),
        SetUpAndRunTool(path_registry=path_instance, object_registry=object_instance),
        ListRegistryObjects(object_registry=object_instance),
        Objects2File(object_registry=object_instance),
        ListRegistryPaths(path_registry=path_instance),
        Paths2File(path_registry=path_instance),
        MapPath2Name(path_registry=path_instance),
        PlanBVisualizationTool(path_registry=path_instance),
        Name2PDBTool(path_registry=path_instance),
        SpecializedCleanTool(path_registry=path_instance),
        RemoveWaterCleaningTool(path_registry=path_instance),
        AddHydrogensCleaningTool(path_registry=path_instance),
    ]

    # add literature search tool
    # Get the api keys
    pqa_key = os.getenv("PQA_API_KEY")
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key))
    return all_tools
