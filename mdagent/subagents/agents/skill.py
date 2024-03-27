import inspect
import json
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from mdagent.utils import PathRegistry

from .prompts import skill_template


class SkillManager:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-3.5-turbo",
        temp=0.1,
        retrieval_top_k=5,
        ckpt_dir="ckpt",
        resume=False,
    ):
        load_dotenv()
        self.dir_name = f"{ckpt_dir}/skill_library"
        self.path_registry = path_registry
        self.retrieval_top_k = retrieval_top_k

        llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_chain = LLMChain(llm=llm, prompt=skill_template)

        self.skills = {}
        # retrieve past skills & tools
        if resume:
            # print(f"\n\033[43mLoading Skills from {self.dir_name}\033[0m")
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

    def _generate_tool_description(self, fxn_code):
        """Given the code snippet, it asks the agent to provide a tool description"""
        return self.llm_chain({"code": fxn_code})["text"]

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

        # Get the parameters of the function
        args = inspect.signature(function).parameters
        arguments = []
        for param in args.values():
            annotation = param.annotation
            param_type = "Any" if annotation == param.empty else str(annotation)
            default_value = None if param.default == param.empty else param.default
            arguments.append(
                {
                    "name": param.name,
                    "type": param_type,
                    "default": default_value,
                }
            )
        self.update_skill_library(function, code, description, arguments)
        return fxn_name

    def update_skill_library(self, function, code_script, description, arguments):
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
            "arguments": arguments,
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

    def execute_skill_function(self, tool_name, **kwargs):
        code = self.skills.get(tool_name, {}).get("code", None)
        if not code:
            raise ValueError(
                f"Code for {tool_name} not found. Make sure to use correct tool name."
                f"Ensure function and args are handed separately."
            )
        # capture initial state
        initial_files = set(os.listdir("."))
        initial_registry = self.path_registry.list_path_names()

        try:
            self._check_arguments(tool_name, **kwargs)
            namespace = {}
            exec(code, namespace)
            function = namespace[tool_name]
            output = function(**kwargs)
        except Exception as e:
            raise e

        # capture final state
        new_files = list(set(os.listdir(".")) - initial_files)
        new_registry = list(
            set(self.path_registry.list_path_names()) - set(initial_registry)
        )
        message = "Successfully executed code."
        if new_files:
            message += f"\nNew Files Created: {', '.join(new_files)}"
        if new_registry:
            message += f"\nFiles added to Path Registry: {', '.join(new_registry)}"
        if output:
            message += f"\nCode Output: {output}\n"
        return message

    def _check_arguments(self, fxn_name, **args):
        expected_args = self.skills.get(fxn_name, {}).get("arguments", {})
        missing_args = []
        for arg in expected_args:
            if arg["name"] not in args:
                missing_args.append(arg["name"])
        if missing_args:
            raise ValueError(f"Missing arguments for {fxn_name}: {missing_args}")

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
