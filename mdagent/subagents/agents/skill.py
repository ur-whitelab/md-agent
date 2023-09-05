import os
import re
import json
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from mdagent.mainagent import _make_llm
from mdagent.subagents.prompts import SkillStep1Prompts, SkillStep2Prompts
from mdagent.tools import PathRegistry


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
        #self.path_registry = path_registry

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)
        self.llm_step1 = self._initialize_llm(SkillStep1Prompts)
        self.llm_step2 = self._initialize_llm(SkillStep2Prompts)

        # retrieve past skills & tools
        if resume:
            print(f"Loading Skills from {ckpt_dir}/skill")
            # retrieve skill library for query
            with open(f"{ckpt_dir}/skill/skills.json", "w") as f1:
                self.skill_library = json.load(f1)

            # retrieve existing learned LangChain tools
            with open(f"{ckpt_dir}/skill/tools.json", "w") as f2:
                self.learnedtools = json.load(f2)
            # OR
            self.learnedtools = self.skills["tools"]
        else:
            self.learnedtools = []
            self.skills = {}

        # to store individual codes - for developers/users to look at
        os.makedirs(f"{ckpt_dir}/skill/code", exist_ok=True)

        # to store individual tool descriptions - for langchain tools
        os.makedirs(f"{ckpt_dir}/skill/description", exist_ok=True)

        # to store individual tools - for langchain tools
        os.makedirs(f"{ckpt_dir}/skill/langchain_tool", exist_ok=True)

    def _create_prompt(self, prompts):
        suffix = ""
        human_prompt = PromptTemplate(
            template=prompts.PROMPT,
            input_variables=["code"],
        )

        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([prompts.PREFIX, prompts.FORMAT])
        )

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )

    def _initialize_llm(self,prompts):
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(prompts),
            callback_manager=StreamingStdOutCallbackHandler,
        )
        return llm_chain
    
    def generate_tool_description_step1(self, fxn_code):
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

    def generate_full_code_step2(self, code, info):
        fxn_name = info["fxn_name"]
        tool_name = info["tool_name"]
        description = info["description"]
        response = self.llm_step2(
            {"code": code, 
            "fxn_name": fxn_name, 
            "tool_name": tool_name, 
            "description": description}
        )
        match = re.search(r'Full Code: (.+)', response, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def add_new_tool(self, code):
        info = self.generate_tool_description_step1(code)
        if info == None:
            print("Skill agent failed to provide tool description.")
            return None
        print(f"Skill agent generated tool description for the new tool.")
        full_code = self.generate_full_code_step2(code, info)
        if full_code == None:
            print("Skill agent failed to provide full code.")
            return None
        print(f"Skill agent generated full code for the new LangChain tool.")
        tool_name = info["tool_name"]
        description = info["tool_description"]
        if tool_name in self.learnedtools: # imperfect way to check
            print(f"Tool with similar name already exists: {tool_name}. Rewriting!")
            i = 2
            while f"{tool_name}V{i}.py" in os.listdir(f"{self.ckpt_dir}/skill/code"):
                i += 1
            filename = f"{tool_name}V{i}.py"
        else:
            filename = tool_name

        with open(f"{self.ckpt_dir}/skill/code/{filename}", "w") as f0:
            f0.write(code)

        with open(f"{self.ckpt_dir}/skill/description/{filename}", "w") as f1:
            f1.write(description)

        with open(f"{self.ckpt_dir}/skill/langchain_tool/{filename}", "w") as f2:
            f2.write(full_code)

        self.learnedtools.append(tool_name)
        self.skills[tool_name] = {
            "code": code,
            "description": description,
            "full_code": full_code,
        }
        return tool_name

    def run(self, code, max_retries=3):
        for retry in range(max_retries):
            tool_name = self.add_new_tool(code)
            if tool_name:
                return tool_name
        print(f"Skill agent failed to add a new tool {max_retries} times. Move on.")
        return None
        


        

