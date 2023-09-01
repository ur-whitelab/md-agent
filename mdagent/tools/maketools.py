import os

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel

from .base_tools import (
    AddHydrogensCleaningTool,
    CheckDirectoryFiles,
    ListRegistryPaths,
    MapPath2Name,
    Name2PDBTool,
    PathRegistry,
    PlanBVisualizationTool,
    RemoveWaterCleaningTool,
    Scholar2ResultLLM,
    SetUpAndRunTool,
    SpecializedCleanTool,
    VisualizationToolRender,
)
from .subagent_tools import ToolCreation, SkillUpdate, SkillQuery

def make_tools(llm: BaseLanguageModel, subagents, verbose=False):
    load_dotenv()

    all_tools = agents.load_tools(["python_repl", "human", "llm-math"], llm)
        
    # add registry tools
    path_instance = PathRegistry.get_instance() # get instance first
    
    # add base tools
    base_tools = [
        VisualizationToolRender(),
        CheckDirectoryFiles(),
        SetUpAndRunTool(path_registry=path_instance),
        ListRegistryPaths(path_registry=path_instance),
        MapPath2Name(path_registry=path_instance),
        PlanBVisualizationTool(path_registry=path_instance),
        Name2PDBTool(path_registry=path_instance),
        SpecializedCleanTool(path_registry=path_instance),
        RemoveWaterCleaningTool(path_registry=path_instance),
        AddHydrogensCleaningTool(path_registry=path_instance),
    ]

    # base tools using sub agents
    subagents_tools = [
        ToolCreation(subagents, path_registry=path_instance),
        SkillUpdate(subagents.skill_agent,path_registry=path_instance),
        SkillQuery(subagents.skill_agent, path_registry=path_instance),
    ]

    # add 'learned' tools here
    learned_tools = [
        # load from pickle file or VDB
    ]

    all_tools += base_tools + subagents_tools + learned_tools

    # add other tools depending on api keys
    pqa_key = os.getenv("PQA_API_KEY")
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key)) # literature search


    return all_tools