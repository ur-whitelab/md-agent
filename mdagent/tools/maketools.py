import os

import streamlit as st
from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from mdagent.utils import PathRegistry

from .base_tools import (
    CleaningToolFunction,
    ComputeAcylindricity,
    ComputeAsphericity,
    ComputeDSSP,
    ComputeGyrationTensor,
    ComputeLPRMSD,
    ComputeRelativeShapeAntisotropy,
    ComputeRMSD,
    ComputeRMSF,
    ContactsTool,
    DistanceMatrixTool,
    GetActiveSites,
    GetAllKnownSites,
    GetAllSequences,
    GetBindingSites,
    GetGeneNames,
    GetInteractions,
    GetKineticProperties,
    GetPDB3DInfo,
    GetPDBProcessingInfo,
    GetProteinAssociatedKeywords,
    GetProteinFunction,
    GetRelevantSites,
    GetSequenceInfo,
    GetSubunitStructure,
    GetTurnsBetaSheetsHelices,
    GetUniprotID,
    ListRegistryPaths,
    MapProteinRepresentation,
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
    SummarizeProteinStructure,
    UniprotID2Name,
    VisualizeProtein,
)


def make_all_tools(
    llm: BaseLanguageModel,
    human=False,
):
    load_dotenv()
    all_tools = []
    path_instance = PathRegistry.get_instance()  # get instance first
    if llm:
        all_tools += agents.load_tools(["llm-math"], llm)
        # all_tools += [PythonREPLTool()]
        all_tools += [
            ModifyBaseSimulationScriptTool(path_registry=path_instance, llm=llm),
        ]
        if "OPENAI_API_KEY" in os.environ:
            all_tools += [Scholar2ResultLLM(llm=llm, path_registry=path_instance)]
        if human:
            all_tools += [agents.load_tools(["human"], llm)[0]]

    # add base tools
    base_tools = [
        SummarizeProteinStructure(path_registry=path_instance),
        ComputeAcylindricity(path_registry=path_instance),
        ComputeAsphericity(path_registry=path_instance),
        ComputeDSSP(path_registry=path_instance),
        ComputeGyrationTensor(path_registry=path_instance),
        ComputeRelativeShapeAntisotropy(path_registry=path_instance),
        CleaningToolFunction(path_registry=path_instance),
        ComputeLPRMSD(path_registry=path_instance),
        ComputeRMSD(path_registry=path_instance),
        ComputeRMSF(path_registry=path_instance),
        ContactsTool(path_registry=path_instance),
        DistanceMatrixTool(path_registry=path_instance),
        ListRegistryPaths(path_registry=path_instance),
        MomentOfInertia(path_registry=path_instance),
        PackMolTool(path_registry=path_instance),
        PCATool(path_registry=path_instance),
        PPIDistance(path_registry=path_instance),
        ProteinName2PDBTool(path_registry=path_instance),
        RadiusofGyrationAverage(path_registry=path_instance),
        RadiusofGyrationPerFrame(path_registry=path_instance),
        RadiusofGyrationPlot(path_registry=path_instance),
        RDFTool(path_registry=path_instance),
        SetUpandRunFunction(path_registry=path_instance),
        SimulationOutputFigures(path_registry=path_instance),
        SmallMolPDB(path_registry=path_instance),
        SolventAccessibleSurfaceArea(path_registry=path_instance),
        VisualizeProtein(path_registry=path_instance),
        MapProteinRepresentation(),
        UniprotID2Name(),
        GetBindingSites(),
        GetActiveSites(),
        GetRelevantSites(),
        GetAllKnownSites(),
        GetProteinFunction(),
        GetProteinAssociatedKeywords(),
        GetAllSequences(),
        GetInteractions(),
        GetSubunitStructure(),
        GetSequenceInfo(),
        GetPDBProcessingInfo(),
        GetPDB3DInfo(),
        GetTurnsBetaSheetsHelices(),
        GetUniprotID(),
        GetGeneNames(),
        GetKineticProperties(),
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
