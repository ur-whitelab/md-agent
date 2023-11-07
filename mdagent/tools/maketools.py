import json
import os
from typing import List, Optional, Type

from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import BaseTool, StructuredTool
from langchain.vectorstores import Chroma
from pydantic import BaseModel, Field

from mdagent.subagents import Iterator, SubAgentSettings
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
from .subagent_tools import ExecuteSkill, SkillRetrieval


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
):
    load_dotenv()
    all_tools = []

    if llm:
        # all_tools += agents.load_tools(["python_repl", "human", "llm-math"], llm)
        all_tools += agents.load_tools(["python_repl", "llm-math"], llm)

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
        # where is plotting tool?
    ]

    # tools using subagents
    if subagent_settings is None:
        subagent_settings = SubAgentSettings(path_registry=path_instance)
    subagents_tools = []
    if not skip_subagents:
        subagents_tools = [
            CreateNewTool(
                path_registry=path_instance, subagent_settings=subagent_settings
            ),
            ExecuteSkill(
                path_registry=path_instance, subagent_settings=subagent_settings
            ),
            SkillRetrieval(
                path_registry=path_instance, subagent_settings=subagent_settings
            ),
        ]

    # add 'learned' tools here
    # disclaimer: assume they don't need path_registry
    learned_tools = []
    if subagent_settings.resume:
        learned_tools = get_learned_tools(subagent_settings.ckpt_dir)

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
    subagents_required=True,
):
    retrieved_tools = []
    if subagents_required:
        # add subagents-related tools by default
        path_instance = PathRegistry.get_instance()
        retrieved_tools = [
            CreateNewTool(
                path_registry=path_instance, subagent_settings=subagent_settings
            ),
            ExecuteSkill(
                path_registry=path_instance, subagent_settings=subagent_settings
            ),
            SkillRetrieval(
                path_registry=path_instance, subagent_settings=subagent_settings
            ),
        ]
        retrieval_top_k -= len(retrieved_tools)
        all_tools = make_all_tools(llm, subagent_settings, skip_subagents=True)
    else:
        all_tools = make_all_tools(llm, subagent_settings, skip_subagents=False)

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
    k = min(retrieval_top_k, vectordb._collection.count())
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


class CreateNewToolInputs(BaseModel):
    """Input for Create New Tool"""

    task: str = Field(..., description="Description of task the tool should perform")
    orig_prompt: str = Field(..., description="Full user prompt from the beginning")
    curr_tools: List[str] = Field(..., description="List of tools the agent has")


class CreateNewTool(BaseTool):
    name = "CreateNewTool"
    description = """
        This tool is used to create a new tool.
        Given a description of the tool needed,
        it will write and test tools.
        If this tool hits maximum iterations without suceeding
        to create a tool, it will return a failure. If you
        receive a failure, you can try again with a different
        input description. If you receive a success, you will
        recieve the tool name, description, and input type.
        You can then use the tool in subsequent steps.

        Args:
            task (str): Description of task the tool should perform
            orig_prompt (str): Full user prompt from the beginning
            curr_tools (List[str]): List of tools the agent has
    """
    arg_schema: Type[BaseModel] = CreateNewToolInputs
    path_registry: Optional[PathRegistry]
    subagent_settings: Optional[SubAgentSettings]

    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        subagent_settings: Optional[SubAgentSettings],
    ):
        super().__init__()
        self.path_registry = path_registry
        self.subagent_settings = subagent_settings

    def get_all_tools_string(self):
        llm = _make_llm(model="gpt-3.5-turbo", temp=0.1, verbose=True)
        all_tools = make_all_tools(llm, self.subagent_settings, skip_subagents=True)
        all_tools_string = ""
        for tool in all_tools:
            all_tools_string += f"{tool.name}: {tool.description}\n"
        return all_tools_string

    # def _run(self, task: str, orig_prompt: str, curr_tools: List[str]) -> str:
    def _run(self, task: str) -> str:
        """use the tool."""
        # check formatting
        try:
            if self.path_registry is None:
                return "Path registry not initialized"
            if self.subagent_settings is None:
                return "Settings for subagents yet to be defined"
            # run iterator
            print("getting all tools info")
            all_tools_string = self.get_all_tools_string()
            print("setting up iterator")
            newcode_iterator = Iterator(
                self.path_registry,
                self.subagent_settings,
                all_tools_string=all_tools_string,
                # current_tools=curr_tools,
            )
            print("running iterator")
            # tool_name = newcode_iterator.run(task, orig_prompt)
            tool_name = newcode_iterator.run(task, task)
            if tool_name:
                return f"""Tool created successfully: {tool_name}
                You can now use the tool in subsequent steps."""
            else:
                return "The 'CreateNewTool' tool failed to build a new tool."
        except Exception as e:
            return f"Something went wrong. {type(e).__name__}: {e}"

    async def _arun(self, task: str, orig_prompt: str, curr_tools: List[str]) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
