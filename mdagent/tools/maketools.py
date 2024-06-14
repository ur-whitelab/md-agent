import streamlit as st
from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from mdagent.utils import PathRegistry

from .base_tools import (
    CleaningToolFunction,
    ComputeLPRMSD,
    ComputeRMSD,
    ComputeRMSF,
    ContactsTool,
    DistanceMatrixTool,
    ListRegistryPaths,
    ModifyBaseSimulationScriptTool,
    MomentOfInertia,
    PackMolTool,
    PCATool,
    PPIDistance,
    ProteinName2PDBTool,
    RadiusofGyrationAverage,
    RadiusofGyrationPerFrame,
    RadiusofGyrationPlot,
    RDFTool,
    Scholar2ResultLLM,
    SetUpandRunFunction,
    SimulationOutputFigures,
    SmallMolPDB,
    SolventAccessibleSurfaceArea,
    VisualizeProtein,
)


def make_all_tools(
    llm: BaseLanguageModel,
    human=False,
    callbacks=None,
):
    load_dotenv()
    all_tools = []
    path_instance = PathRegistry.get_instance()  # get instance first
    if llm:
        all_tools += agents.load_tools(["llm-math"], llm)
        # all_tools += [PythonREPLTool()]
        all_tools += [
            ModifyBaseSimulationScriptTool(
                path_registry=path_instance, callbacks=callbacks, llm=llm
            ),
            Scholar2ResultLLM(
                llm=llm, path_registry=path_instance, callbacks=callbacks
            ),
        ]
        if human:
            all_tools += [agents.load_tools(["human"], llm)[0]]

    # add base tools
    base_tools = [
        CleaningToolFunction(path_registry=path_instance, callbacks=callbacks),
        ComputeLPRMSD(path_registry=path_instance, callbacks=callbacks),
        ComputeRMSD(path_registry=path_instance, callbacks=callbacks),
        ComputeRMSF(path_registry=path_instance, callbacks=callbacks),
        ContactsTool(path_registry=path_instance, callbacks=callbacks),
        DistanceMatrixTool(path_registry=path_instance, callbacks=callbacks),
        ListRegistryPaths(path_registry=path_instance, callbacks=callbacks),
        MomentOfInertia(path_registry=path_instance, callbacks=callbacks),
        PackMolTool(path_registry=path_instance, callbacks=callbacks),
        PCATool(path_registry=path_instance, callbacks=callbacks),
        PPIDistance(path_registry=path_instance, callbacks=callbacks),
        ProteinName2PDBTool(path_registry=path_instance, callbacks=callbacks),
        RadiusofGyrationAverage(path_registry=path_instance, callbacks=callbacks),
        RadiusofGyrationPerFrame(path_registry=path_instance, callbacks=callbacks),
        RadiusofGyrationPlot(path_registry=path_instance, callbacks=callbacks),
        RDFTool(path_registry=path_instance, callbacks=callbacks),
        SetUpandRunFunction(path_registry=path_instance, callbacks=callbacks),
        SimulationOutputFigures(path_registry=path_instance, callbacks=callbacks),
        SmallMolPDB(path_registry=path_instance, callbacks=callbacks),
        SolventAccessibleSurfaceArea(path_registry=path_instance, callbacks=callbacks),
        VisualizeProtein(path_registry=path_instance, callbacks=callbacks),
    ]

    all_tools += base_tools
    return all_tools


def get_tools(
    query,
    llm: BaseLanguageModel,
    top_k_tools=15,
    human=False,
):
    ckpt_dir = PathRegistry.get_instance().ckpt_dir

    all_tools = make_all_tools(llm, human=human)

    # set vector DB for all tools
    vectordb = Chroma(
        collection_name="all_tools_vectordb",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=f"{ckpt_dir}/all_tools_vectordb",
    )
    # vectordb.delete_collection()      #<--- to clear previous vectordb directory
    for i, tool in enumerate(all_tools):
        vectordb.add_texts(
            texts=[tool.description],
            ids=[tool.name],
            metadatas=[{"tool_name": tool.name, "index": i}],
        )
        vectordb.persist()

    # retrieve 'k' tools
    k = min(top_k_tools, vectordb._collection.count())
    if k == 0:
        return None
    docs = vectordb.similarity_search(query, k=k)
    retrieved_tools = []
    for d in docs:
        index = d.metadata.get("index")
        if index is not None and 0 <= index < len(all_tools):
            retrieved_tools.append(all_tools[index])
        else:
            print(f"Invalid index {index}.")
            print("Some tools may be duplicated.")
            print(f"Try to delete vector DB at {ckpt_dir}/all_tools_vectordb.")
            st.markdown(
                "Invalid index. Some tools may be duplicated Try to delete VDB.",
                unsafe_allow_html=True,
            )
    return retrieved_tools
