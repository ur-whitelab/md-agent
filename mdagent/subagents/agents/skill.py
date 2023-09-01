import json
import os

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

from ..mdagent.agent import _make_llm
from ..mdagent.prompts import SkillPrompts


class Skill:
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
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(),
            callback_manager=StreamingStdOutCallbackHandler,
        )

        # retrieve past skills & tools
        if resume:
            print(f"\033[33mLoading Skills from {ckpt_dir}/skill\033[0m")
            # retrieve skill library for query
            with open(f"{ckpt_dir}/skill/skills.json", "w") as f1:
                self.skill_library = json.load(f1)

            # retrieve existing learned LangChain tools
            with open(f"{ckpt_dir}/skill/tools.json", "w") as f2:
                self.learnedtools = json.load(f2)
            # OR
            self.learnedtools = self.skill_library["tools"]
        else:
            self.learnedtools = []
            self.skill_library = {}

        # to store individual codes - for developers/users to look at
        os.makedirs(f"{ckpt_dir}/skill/code", exist_ok=True)

        # to store individual tool descriptions - for langchain tools?
        os.makedirs(f"{ckpt_dir}/skill/description", exist_ok=True)

        # to store individual tools - for langchain tools?
        # can remove this after developing is done
        os.makedirs(f"{ckpt_dir}/skill/tools", exist_ok=True)

    def _create_prompt(self):
        suffix = ""
        human_prompt = PromptTemplate(
            template=SkillPrompts.PROMPT,
            input_variables=["code"],
        )

        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([SkillPrompts.PREFIX, SkillPrompts.FORMAT])
        )

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )

    def run(self, code):
        output = self.llm_chain({"code", code})
        return output

    # # skill library for skill query (goal: to provide code) (?)
    # @property
    # def skill_library_string(self):
    #     """
    #     this 'packs' all tools into a string and pass it to other agents
    #     (curriculum agent, for one)
    #     """
    #     toolbox = ""
    #     for tool_name, entry in self.learnedtools.items():
    #         toolbox += f"{entry['code']}\n\n"
    #     for tool in self.base_tools: # won't work for langchain BaseTool object
    #         toolbox += f"{tool}\n\n"
    #     return toolbox
