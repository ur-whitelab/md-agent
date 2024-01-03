import json
import os
from typing import Optional, Type

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import BaseTool, StructuredTool
from langchain.vectorstores import Chroma
from langchain_experimental.tools import PythonREPLTool
from pydantic import BaseModel, Field

from mdagent.subagents import Iterator, SubAgentSettings
from mdagent.utils import PathRegistry, _make_llm

from .base_tools import (
    CheckDirectoryFiles,
    CleaningToolFunction,
    ListRegistryPaths,
    MapPath2Name,
    ModifyBaseSimulationScriptTool,
    Name2PDBTool,
    PackMolTool,
    PPIDistance,
    RMSDCalculator,
    Scholar2ResultLLM,
    SerpGitTool,
    SetUpandRunFunction,
    SimulationOutputFigures,
    VisualizeProtein,
)
from .subagent_tools import ExecuteSkill, SkillRetrieval, WorkflowPlan


def get_learned_tools(ckpt_dir="ckpt"):
    skill_file_path = f"{ckpt_dir}/skill_library/skills.json"
    if os.path.exists(skill_file_path):
        with open(skill_file_path, "r") as f1:
            content = f1.read().strip()
            if content:
                skills = json.loads(content)
    else:
        raise FileNotFoundError(
            f"Could not find learned tools at {skill_file_path}."
            "Please check your 'ckpt_dir' path."
        )
    learned_tools = []
    for key in skills:
        fxn_name = key
        code = skills[fxn_name]["code"]
        namespace = {}
        exec(code, namespace)
        function = namespace[fxn_name]
        learned_tools.append(StructuredTool.from_function(func=function))
    return learned_tools


def make_all_tools(
    llm: BaseLanguageModel,
    subagent_settings: Optional[SubAgentSettings] = None,
    skip_subagents=False,
    human=False,
):
    load_dotenv()
    all_tools = []

    if llm:
        all_tools += agents.load_tools(["llm-math"], llm)
        all_tools += [PythonREPLTool()]  # or PythonREPLTool(llm=llm)?
        if human:
            all_tools += [agents.load_tools(["human"], llm)[0]]

    # get path registry
    path_instance = PathRegistry.get_instance()  # get instance first

    # add base tools
    base_tools = [
        CleaningToolFunction(path_registry=path_instance),
        CheckDirectoryFiles(),
        ListRegistryPaths(path_registry=path_instance),
        MapPath2Name(path_registry=path_instance),
        Name2PDBTool(path_registry=path_instance),
        PackMolTool(path_registry=path_instance),
        VisualizeProtein(path_registry=path_instance),
        PPIDistance(),
        RMSDCalculator(),
        SetUpandRunFunction(path_registry=path_instance),
        ModifyBaseSimulationScriptTool(path_registry=path_instance, llm=llm),
        SimulationOutputFigures(),
    ]

    # tools using subagents
    if subagent_settings is None:
        subagent_settings = SubAgentSettings(path_registry=path_instance)
    subagents_tools = []
    if not skip_subagents:
        subagents_tools = [
            CreateNewTool(subagent_settings=subagent_settings),
            ExecuteSkill(subagent_settings=subagent_settings),
            SkillRetrieval(subagent_settings=subagent_settings),
            WorkflowPlan(subagent_settings=subagent_settings),
        ]

    # add 'learned' tools here
    # disclaimer: assume they don't need path_registry
    learned_tools = []
    if subagent_settings.resume:
        learned_tools = get_learned_tools(subagent_settings.ckpt_dir)

    all_tools += base_tools + subagents_tools + learned_tools

    # add other tools depending on api keys
    serp_key = os.getenv("SERP_API_KEY")
    pqa_key = os.getenv("PQA_API_KEY")
    if serp_key:
        all_tools.append(SerpGitTool(serp_key))  # github issues search
    if pqa_key:
        all_tools.append(Scholar2ResultLLM(pqa_key))  # literature search
    return all_tools


def get_tools(
    query,
    llm: BaseLanguageModel,
    subagent_settings: Optional[SubAgentSettings] = None,
    top_k_tools=15,
    subagents_required=True,
    human=False,
):
    if subagent_settings:
        ckpt_dir = subagent_settings.ckpt_dir
    else:
        ckpt_dir = "ckpt"

    retrieved_tools = []
    if subagents_required:
        # add subagents-related tools by default
        retrieved_tools = [
            CreateNewTool(subagent_settings=subagent_settings),
            ExecuteSkill(subagent_settings=subagent_settings),
            SkillRetrieval(subagent_settings=subagent_settings),
            WorkflowPlan(subagent_settings=subagent_settings),
        ]
        top_k_tools -= len(retrieved_tools)
        all_tools = make_all_tools(
            llm, subagent_settings, skip_subagents=True, human=human
        )
    else:
        all_tools = make_all_tools(
            llm, subagent_settings, skip_subagents=False, human=human
        )

    # create vector DB for all tools
    vectordb = Chroma(
        collection_name="all_tools_vectordb",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=f"{ckpt_dir}/all_tools_vectordb",
    )
    # vectordb.delete_collection() # to clear vectordb directory
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
    for d in docs:
        index = d.metadata.get("index")
        if index is not None and 0 <= index < len(all_tools):
            retrieved_tools.append(all_tools[index])
        else:
            print(f"Invalid index {index}.")
            print(f"Try deleting vectordb at {ckpt_dir}/all_tools_vectordb.")
    return retrieved_tools


class CreateNewToolInputSchema(BaseModel):
    task: str = Field(description="Description of task the tool should perform.")
    orig_prompt: str = Field(description="Full user prompt you got from the beginning.")
    curr_tools: str = Field(
        description="""List of all tools you have access to. Such as
        this tool, 'ExecuteSkill', 'SkillRetrieval', and maybe `Name2PDBTool`, etc."""
    )


class CreateNewTool(BaseTool):
    name: str = "CreateNewTool"
    description: str = """
        Only use if you don't have right tools for sure and need a different tool.
        If succeeded, it will return the name of the tool.
        You can then use the tool in subsequent steps.
    """
    args_schema: Type[BaseModel] = CreateNewToolInputSchema
    subagent_settings: Optional[SubAgentSettings]

    def __init__(self, subagent_settings: Optional[SubAgentSettings] = None):
        super().__init__()
        self.subagent_settings = subagent_settings

    def get_all_tools_string(self):
        llm = _make_llm(model="gpt-3.5-turbo", temp=0.1, verbose=True)
        all_tools = make_all_tools(llm, self.subagent_settings, skip_subagents=True)
        all_tools_string = ""
        for tool in all_tools:
            all_tools_string += f"{tool.name}: {tool.description}\n"
        return all_tools_string

    def _run(self, task, orig_prompt, curr_tools):
        # def _run(self, task, orig_prompt):
        try:
            # run iterator
            path_registry = self.subagent_settings.path_registry
            print("getting all tools info")
            all_tools_string = self.get_all_tools_string()
            print("setting up iterator")
            newcode_iterator = Iterator(
                path_registry,
                self.subagent_settings,
                all_tools_string=all_tools_string,
                current_tools=curr_tools,
            )
            print("running iterator")
            tool_name = newcode_iterator.run(task, orig_prompt)
            # tool_name = newcode_iterator.run(task, task)
            if tool_name:
                return f"""Tool created successfully: {tool_name}
                You can now use the tool in subsequent steps."""
            else:
                return "The 'CreateNewTool' tool failed to build a new tool."
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
