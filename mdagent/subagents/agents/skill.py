import json
import os
import re

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
from mdagent.utils import _make_llm


class SkillAgent:
    def __init__(
        self,
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

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)
        self.llm_step1 = self._initialize_llm(SkillStep1Prompts)
        self.llm_step2 = self._initialize_llm(SkillStep2Prompts)

        # retrieve past skills & tools
        if resume:
            print(f"Loading Skills from {ckpt_dir}/skill_library")
            # retrieve skill library for query
            with open(f"{ckpt_dir}/skill_library/skills.json", "w") as f1:
                self.skills = json.load(f1)
        else:
            self.skills = {}

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
        response = self.llm_step1({"code": fxn_code})
        fxn_name_match = re.search(r"Function name:\s*(\w+)", response)
        tool_name_match = re.search(r"Tool name:\s*(\w+)", response)
        description_match = re.search(r"Description:\s*(.*)", response, re.DOTALL)

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
        )
        match = re.search(r"Full Code: (.+)", response, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def add_new_tool(self, code, max_retries=3):
        retry = 0
        tool_info = None
        full_code = None
        while retry <= max_retries:
            # step 1: generate tool description & tool name
            if tool_info is None:
                tool_info = self._generate_tool_description_step1(code)
                if tool_info is None:
                    print("Skill agent failed to provide tool description. Retrying...")
                    retry += 1
                    continue
                print("Tool description for the new tool is successfully created.")

            # step 2: generate the full code script for new LangChain tool
            full_code = self._generate_full_code_step2(code, tool_info)
            if full_code is None:
                print("Skill agent failed to provide full code. Retrying...")
                retry += 1
                continue
            else:
                print("Full code for the new LangChain tool is created.")
                break
        if full_code is None:
            print(
                f"""Skill agent failed to add a new tool {max_retries}
            times. Saved the code and move on."""
            )
            with open(f"{self.ckpt_dir}/skill_library/failed.txt", "a") as f:
                f.write("\n\nFAILED TO ADD THE CODE BELOW AS A NEW TOOL: \n")
                f.write(code)
            return None

        # add successful full code to skill library
        tool_name = tool_info["tool_name"]
        description = tool_info["tool_description"]
        if tool_name in self.skills:  # imperfect way to check
            print(f"Tool with similar name already exists: {tool_name}. Rewriting!")
            i = 2
            while f"{tool_name}V{i}.py" in os.listdir(
                f"{self.ckpt_dir}/skill_library/code"
            ):
                i += 1
            filename = f"{tool_name}V{i}.py"
        else:
            filename = tool_name

        self.skills[tool_name] = {
            "code": code,
            "description": description,
            "full_code": full_code,
        }
        with open(f"{self.ckpt_dir}/skill_library/code/{filename}", "w") as f0:
            f0.write(code)

        with open(f"{self.ckpt_dir}/skill_library/description/{filename}", "w") as f1:
            f1.write(description)

        with open(
            f"{self.ckpt_dir}/skill_library/langchain_tool/{filename}", "w"
        ) as f2:
            f2.write(full_code)

        with open(f"{self.ckpt_dir}/skill_library/skills.json", "w") as f3:
            json.dump(self.skills, f3)

        return tool_name

    def get_skills(self):
        return self.skills

    def run(self, code, max_retries=3):
        self.add_new_tool(code, max_retries)
