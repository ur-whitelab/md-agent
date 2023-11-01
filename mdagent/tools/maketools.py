import os
import pickle
from typing import Optional

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from mdagent.subagents import SubAgentSettings
from mdagent.utils import PathRegistry, _make_llm

from .base_tools.clean_tools import (
    AddHydrogensCleaningTool,
    RemoveWaterCleaningTool,
    SpecializedCleanTool,
)
from .base_tools.md_util_tools import Name2PDBTool
from .base_tools.registry_tools import ListRegistryPaths, MapPath2Name
from .base_tools.search_tools import Scholar2ResultLLM
from .base_tools.setup_and_run import SetUpAndRunTool
from .base_tools.vis_tools import (
    CheckDirectoryFiles,
    PlanBVisualizationTool,
    VisualizationToolRender,
)
from .subagent_tools import CreateNewTool, ExecuteSkill, SkillRetrieval


def get_all_tools_string():
    llm = _make_llm(model="gpt-3.5-turbo", temp=0.1, verbose=True)
    all_tools = make_all_tools(llm)
    all_tools_string = ""
    for tool in all_tools:
        all_tools_string += f"{tool.name}: {tool.description}\n"
    return all_tools_string


def make_all_tools(
    llm: BaseLanguageModel, subagent_settings: Optional[SubAgentSettings] = None
):
    load_dotenv()
    all_tools = []

    if llm:
        all_tools += agents.load_tools(["python_repl", "human", "llm-math"], llm)

    # get path registry
    path_instance = PathRegistry.get_instance()  # get instance first

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
    if subagent_settings is None:
        subagent_settings = SubAgentSettings(path_registry=path_instance)
    subagents_tools = [
        CreateNewTool(path_registry=path_instance, subagent_settings=subagent_settings),
        ExecuteSkill(path_registry=path_instance, subagent_settings=subagent_settings),
        SkillRetrieval(
            path_registry=path_instance, subagent_settings=subagent_settings
        ),
    ]

    # add 'learned' tools here
    # disclaimer: assume every learned tool has path_registry as an argument
    learned_tools = []
    if subagent_settings.resume:
        pickle_file = f"{subagent_settings.ckpt_dir}/skill_library/langchain_tools.pkl"
        if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
            with open(pickle_file, "rb") as f:
                loaded_tools = pickle.load(f)
            for tool in loaded_tools:
                learned_tools.append(tool(path_registry=path_instance))

    all_tools += base_tools + subagents_tools + learned_tools

    # add other tools depending on api keys
    pqa_key = os.getenv("PQA_API_KEY")
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key))  # literature search

    return all_tools


def get_tools(
    query,
    llm: BaseLanguageModel,
    subagent_settings: Optional[SubAgentSettings] = None,
    ckpt_dir="ckpt",
    retrieval_top_k=10,
):
    all_tools = make_all_tools(llm, subagent_settings)

    # create vector DB for all tools
    vectordb = Chroma(
        collection_name="all_tools_vectordb",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=f"{ckpt_dir}/all_tools_vectordb",
    )
    for i, tool in enumerate(all_tools):
        vectordb.add_texts(
            texts=[tool.description],
            ids=[tool.name],
            metadatas=[{"tool_name": tool.name, "index": i}],
        )
        vectordb.persist()

    # retrieve 'k' tools
    k = min(retrieval_top_k, vectordb._collection.count())
    if k == 0:
        return None
    docs = vectordb.similarity_search(query, k=k)
    retrieved_tools = []
    for d in docs:
        retrieved_tools.append(all_tools[d.metadata["index"]])
    return retrieved_tools
