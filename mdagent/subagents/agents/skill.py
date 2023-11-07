import json
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import Chroma

from mdagent.subagents.prompts import SkillPrompts
from mdagent.utils import PathRegistry, _make_llm


class SkillManager:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-3.5",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        retrieval_top_k=5,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
    ):
        load_dotenv()
        self.dir_name = f"{ckpt_dir}/skill_library"
        self.path_registry = path_registry
        self.retrieval_top_k = retrieval_top_k

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)

        self.skills = {}
        # retrieve past skills & tools
        if resume:
            print(f"\n\033[43mLoading Skills from {self.dir_name}\033[0m")
            skill_file_path = f"{self.dir_name}/skills.json"
            if os.path.exists(skill_file_path):
                with open(skill_file_path, "r") as f1:
                    content = f1.read().strip()
                    if content:
                        self.skills = json.loads(content)
                    else:
                        print(f"\033[43mSkill file {skill_file_path} is empty\033[0m")
            else:
                print(f"\033[43mNo skill file found at {skill_file_path}\033[0m")

        os.makedirs(f"{self.dir_name}/code", exist_ok=True)
        os.makedirs(f"{self.dir_name}/description", exist_ok=True)
        os.makedirs(f"{self.dir_name}/vectordb", exist_ok=True)
        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/skill_library/vectordb",
        )

    def get_skills(self):
        return self.skills

    def _create_prompt(self, prompts):
        suffix = ""
        human_prompt = PromptTemplate(
            template=prompts.PROMPT,
            input_variables=prompts.INPUT_VARS,
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([prompts.PREFIX, prompts.FORMAT])
        )
        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )

    def _initialize_llm(self, prompts):
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(prompts),
        )
        return llm_chain

    def _generate_tool_description(self, fxn_code):
        """Given the code snippet, it asks the agent to provide a tool description"""
        llm_chain = self._initialize_llm(SkillPrompts)
        return llm_chain({"code": fxn_code})["text"]

    def add_new_tool(self, fxn_name, code, new_description=False):
        # execute the code to get function
        namespace = {}
        exec(code, namespace)
        function = namespace[fxn_name]

        # get description
        if new_description:  # useful if we need to update description
            description = self._generate_tool_description(code)
        else:
            if function.__doc__ is None:
                description = self._generate_tool_description(code)
            else:
                description = function.__doc__
        self.update_skill_library(function, code, description)
        return fxn_name

    def update_skill_library(self, function, code_script, description):
        fxn_name = function.__name__
        if fxn_name in self.skills:  # TODO: a better way to check for duplicates
            print(f"\n\033[43mFunction {fxn_name} already exists. Rewriting!\033[0m")
            self.vectordb._collection.delete(ids=[fxn_name])
            i = 2
            while f"{fxn_name}V{i}.py" in os.listdir(f"{self.dir_name}/code/"):
                i += 1
            filename = f"{fxn_name}V{i}"
        else:
            filename = fxn_name

        self.skills[fxn_name] = {
            "code": code_script,
            "description": description,
        }
        # store code
        with open(f"{self.dir_name}/code/{filename}.py", "w") as f0:
            f0.write(code_script)
        self.path_registry.map_path(
            name=fxn_name,
            path=f"{self.dir_name}/code/{filename}.py",
            description=f"Code for new tool {fxn_name}",
        )

        # store description - may remove in future
        with open(f"{self.dir_name}/description/{filename}.txt", "w") as f1:
            f1.write(description)

        # update skills.json
        skill_file_path = f"{self.dir_name}/skills.json"
        if os.path.exists(skill_file_path):
            with open(skill_file_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
        existing_data.update(self.skills)
        with open(skill_file_path, "w") as f:
            json.dump(self.skills, f, indent=4)

        # save to vectordb for skill retrieval
        self.vectordb.add_texts(
            texts=[description], ids=[fxn_name], metadatas=[{"name": fxn_name}]
        )
        self.vectordb.persist()

        # # save function to jsonpickle
        # func_file_path = f"{self.dir_name}/functions.json"
        # if os.path.exists(func_file_path):
        #     with open(func_file_path, "r") as f:
        #         existing_data = json.load(f)
        # else:
        #     existing_data = {}
        # existing_data[fxn_name] = jsonpickle.encode(function)

    def execute_skill_function(self, tool_name, path_registry, **kwargs):
        code = self.skills.get(tool_name, {}).get("code", None)
        if not code:
            raise ValueError(
                f"Code for {tool_name} not found. Make sure to use correct tool name."
                f"Ensure function and args are handed separately."
            )
        # capture initial state
        initial_files = set(os.listdir("."))
        initial_registry = path_registry.list_path_names()

        try:
            namespace = {}
            exec(code, namespace)
            function = namespace[tool_name]
            output = function(**kwargs)
        except Exception as e:
            raise e

        # capture final state
        new_files = list(set(os.listdir(".")) - initial_files)
        new_registry = list(
            set(path_registry.list_path_names()) - set(initial_registry)
        )
        success_message = "Successfully executed code."
        files_message = f"New Files Created: {', '.join(new_files)}"
        registry_message = f"Files added to Path Registry: {', '.join(new_registry)}"
        output_message = f"Code Output: {output}"
        return "\n".join(
            [success_message, files_message, registry_message, output_message]
        )

    # def execute_skill_code(self, tool_name, path_registry, **kwargs):
    #     code = self.skills.get(tool_name, {}).get("code", None)
    #     if not code:
    #         raise ValueError(
    #             f"Code for {tool_name} not found. Make sure to use correct tool name."
    #         )
    #     # capture initial state
    #     initial_files = set(os.listdir("."))
    #     initial_registry = path_registry.list_path_names()

    #     # Redirect stdout and stderr to capture the output
    #     original_stdout = sys.stdout
    #     original_stderr = sys.stderr
    #     sys.stdout = captured_stdout = sys.stderr = io.StringIO()
    #     exec_context = {
    #         **kwargs,
    #         **globals(),
    #     }  # spread and set kwargs as variables in env
    #     try:
    #         exec(code, exec_context)
    #         output = captured_stdout.getvalue()
    #     except Exception as e:
    #         # Restore stdout and stderr
    #         sys.stdout = original_stdout
    #         sys.stderr = original_stderr
    #         error_type = type(e).__name__
    #         raise type(e)(f"Error executing code for {tool_name}. {error_type}: {e}")
    #     finally:
    #         # Ensure that stdout and stderr are always restored
    #         sys.stdout = original_stdout
    #         sys.stderr = original_stderr

    #     # capture final state
    #     new_files = list(set(os.listdir(".")) - initial_files)
    #     new_registry = list(
    #         set(path_registry.list_path_names()) - set(initial_registry)
    #     )

    #     success_message = "Successfully executed code."
    #     files_message = f"New Files Created: {', '.join(new_files)}"
    #     registry_message = f"Files added to Path Registry: {', '.join(new_registry)}"
    #     output_message = f"Code Output: {output}"
    #     return "\n".join(
    #         [success_message, files_message, registry_message, output_message]
    #     )

    def retrieve_skills(self, query, k=None):
        if k is None:
            k = self.retrieval_top_k
        k = min(k, self.vectordb._collection.count())
        if k == 0:
            return None
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        retrieved_skills = {}
        string = ""
        for doc, score in docs_and_scores:
            tool_name = doc.metadata["name"]
            retrieved_skills[tool_name] = self.skills[tool_name]["code"]
            string += tool_name + ", Score: " + str(score) + "\n"
        print(f"\n\033[43m{string}\033[0m")
        return retrieved_skills
