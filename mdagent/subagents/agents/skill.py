import importlib.util
import json
import os
import pickle
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from mdagent.subagents.prompts import SkillStep1Prompts, SkillStep2Prompts
from mdagent.utils import PathRegistry, _make_llm


class SkillAgent:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-3.5",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
    ):
        load_dotenv()
        self.ckpt_dir = ckpt_dir
        self.path_registry = path_registry

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)
        self.llm_step1 = self._initialize_llm(SkillStep1Prompts)
        self.llm_step2 = self._initialize_llm(SkillStep2Prompts)

        self.skills = {}
        # retrieve past skills & tools
        if resume:
            print(f"\n\033[42mLoading Skills from {ckpt_dir}/skill_library\033[0m")
            skill_file_path = f"{ckpt_dir}/skill_library/skills.json"
            if os.path.exists(skill_file_path):
                with open(skill_file_path, "r") as f1:
                    content = f1.read().strip()
                    if content:
                        self.skills = json.loads(content)
                    else:
                        print(f"\033[42mSkill file {skill_file_path} is empty\033[0m")
            else:
                print(f"\033[42mNo skill file found at {skill_file_path}\033[0m")

        # to store individual codes - for developers/users to look at
        os.makedirs(f"{ckpt_dir}/skill_library/code", exist_ok=True)

        # to store individual tool descriptions - for langchain tools
        os.makedirs(f"{ckpt_dir}/skill_library/description", exist_ok=True)

        # to store individual tools - for langchain tools
        os.makedirs(f"{ckpt_dir}/skill_library/langchain_tool", exist_ok=True)

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

    def _generate_tool_description_step1(self, fxn_code):
        """
        Given the code snippet, it asks the agent to provide
        1. Python function name
        2. name for Langchain BaseTool
        3. tool description
        """
        response = self.llm_step1({"code": fxn_code})["text"]
        fxn_name_match = re.search(r"Function name:\s*(\w+)", response)
        tool_name_match = re.search(r"Tool name:\s*(\w+)", response)
        description_match = re.search(r"Tool description:\s*(.*)", response, re.DOTALL)

        if fxn_name_match and tool_name_match and description_match:
            fxn_name = fxn_name_match.group(1)
            tool_name = tool_name_match.group(1)
            description = description_match.group(1).strip()
            extracted_info = {
                "fxn_name": fxn_name,
                "tool_name": tool_name,
                "description": description,
            }
            return extracted_info
        else:
            return None

    def _generate_full_code_step2(self, code, info):
        fxn_name = info["fxn_name"]
        tool_name = info["tool_name"]
        description = info["description"]
        response = self.llm_step2(
            {
                "code": code,
                "fxn_name": fxn_name,
                "tool_name": tool_name,
                "description": description,
            }
        )["text"]
        pattern = r"Full Code:\s*```(?:[a-zA-Z]*\s)?(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

    def _dump_tool(self, tool_info, code, full_code):
        tool_name = tool_info["tool_name"]
        description = tool_info["description"]
        if tool_name in self.skills:  # TODO: do a better way to check for duplicates
            print(
                f"\n\033[42mTool with similar name already exists: "
                f"{tool_name}. Rewriting!\033[0m"
            )
            i = 2
            while f"{tool_name}V{i}.py" in os.listdir(
                f"{self.ckpt_dir}/skill_library/code/"
            ):
                i += 1
            filename = f"{tool_name}V{i}.py"
        else:
            filename = tool_name

        # dump codes and description
        with open(f"{self.ckpt_dir}/skill_library/code/{filename}.py", "w") as f0:
            f0.write(code)

        with open(
            f"{self.ckpt_dir}/skill_library/description/{filename}.txt", "w"
        ) as f1:
            f1.write(description)

        tool_path = f"{self.ckpt_dir}/skill_library/langchain_tool/{filename}_tool.py"
        with open(tool_path, "w") as f2:
            f2.write(full_code)

        # save tool to skill library
        self.skills[tool_name] = {
            "code": code,
            "description": description,
            "full_code": full_code,
        }
        skill_file_path = f"{self.ckpt_dir}/skill_library/skills.json"
        if os.path.exists(skill_file_path):
            with open(skill_file_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
        existing_data.update(self.skills)
        with open(skill_file_path, "w") as f3:
            json.dump(self.skills, f3, indent=4)

        # create LangChain BaseTool object by loading it from tool file
        # TODO: move below to iterator code and have critcs check it
        try:
            self._create_LangChain_tool(tool_name, tool_path)
            print(f"LangChain BaseTool object for {tool_name} is successfully created.")
        except Exception as e:
            print(
                f"\n\033[42mFailed to load LangChain tool: {e}"
                "Though the tool is not loaded, the code is saved.\033[0m"
            )

        # add tool file to the registry
        self.path_registry.map_path(
            name=tool_name,
            path=tool_path,
            description=f"Learned tool called {tool_name}",
        )

    def _create_LangChain_tool(self, tool_name, tool_path):
        # load the LangChain BaseTool class from file
        # convert the file path to module name, removing .py and replacing / with .
        module_name = tool_path[:-3].replace("/", ".")

        # load and import the module that holds the tool
        spec = importlib.util.spec_from_file_location(module_name, tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        langchain_tool = getattr(module, tool_name)

        # dump the Langchain BaseTool object to pickle file
        pickle_file = f"{self.ckpt_dir}/skill_library/langchain_tools.pkl"
        if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
            with open(pickle_file, "rb") as f:
                tools = pickle.load(f)
        else:
            tools = []
        tools.append(langchain_tool)
        with open(pickle_file, "wb") as f:
            pickle.dump(tools, f)
        return langchain_tool

    def add_new_tool(self, code, max_retries=3):
        retry = 0
        tool_info = None
        full_code = None
        while retry <= max_retries:
            # step 1: generate tool description & tool name
            if tool_info is None:
                tool_info = self._generate_tool_description_step1(code)
                if tool_info is None:
                    print(
                        "\n\033[42mSkill agent failed to provide tool "
                        "description. Retrying...\033[0m"
                    )
                    retry += 1
                    continue
                print(
                    "\n\033[42mTool description for the new tool is "
                    "successfully created.\033[0m"
                )

            # step 2: generate the full code script for new LangChain tool
            full_code = self._generate_full_code_step2(code, tool_info)
            if full_code is None:
                print(
                    "\n\033[42mSkill agent failed to provide full "
                    "code. Retrying...\033[0m"
                )
                retry += 1
                continue
            else:
                print(
                    "\n\033[42mFull code for the new LangChain tool is created.\033[0m"
                )
                break
        if full_code is None:
            print(
                f"\n\033[42mSkill agent failed to write and add the new tool"
                f" after {max_retries} times. Saved the code and move on.\033[0m"
            )
            with open(f"{self.ckpt_dir}/skill_library/failed.txt", "a") as f:
                f.write("\n\nFAILED TO ADD THE CODE BELOW AS A NEW TOOL: \n")
                f.write(code)
            return None

        # dump successful full code and tool to skill library
        self._dump_tool(tool_info, code, full_code)
        return tool_info["tool_name"]

    def get_skills(self):
        return self.skills

    # TODO: add base tools here
    # def get_base_tools(self):
    #
    #     return [
    #         "VisualizationToolRender",
    #         "CheckDirectoryFiles",
    #         "SetUpAndRunTool",
    #         "ListRegistryPaths",
    #         "MapPath2Name",
    #         "PlanBVisualizationTool",
    #         "Name2PDBTool",
    #         "SpecializedCleanTool",
    #         "RemoveWaterCleaningTool",
    #         "AddHydrogensCleaningTool",
    #     ]

    def run(self, code, max_retries=3):
        self.add_new_tool(code, max_retries)
