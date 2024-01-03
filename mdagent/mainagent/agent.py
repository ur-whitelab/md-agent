from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from mdagent.subagents import SubAgentSettings
from mdagent.utils import PathRegistry, _make_llm

from ..tools import get_tools, make_all_tools
from .prompt import openaifxn_prompt, structured_prompt

load_dotenv()


class AgentType:
    valid_models = {
        "Structured": StructuredChatAgent,
        "OpenAIFunctionsAgent": OpenAIFunctionsAgent,
    }

    @classmethod
    def get_agent(cls, model_name: str = "OpenAIFunctionsAgent"):
        try:
            agent = cls.valid_models[model_name]
            return agent
        except KeyError:
            raise ValueError(
                f"""Invalid agent type: {model_name}
                Please choose from {cls.valid_models.keys()}"""
            )


class MDAgent:
    def __init__(
        self,
        tools=None,
        agent_type="OpenAIFunctionsAgent",  # this can also be strucured_chat
        model="gpt-4-1106-preview",  # current name for gpt-4 turbo
        tools_model="gpt-4-1106-preview",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        path_registry=None,
        subagents_model="gpt-4-1106-preview",
        ckpt_dir="ckpt",
        resume=False,
        top_k_tools=20,  # set "all" if you want to use all tools (& skills if resume)
        use_human_tool=False,
    ):
        if path_registry is None:
            path_registry = PathRegistry.get_instance()
        self.agent_type = agent_type
        self.tools = tools
        self.tools_llm = _make_llm(tools_model, temp, verbose)
        self.use_human_tool = use_human_tool
        self.top_k_tools = top_k_tools

        self.llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        # assign prompt
        if agent_type == "Structured":
            self.prompt = structured_prompt
        elif agent_type == "OpenAIFunctionsAgent":
            self.prompt = openaifxn_prompt

        self.subagents_settings = SubAgentSettings(
            path_registry=path_registry,
            subagents_model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            verbose=verbose,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )

    def _initialize_tools_and_agent(self, user_input=None):
        if self.tools is not None:
            tools = self.tools
        elif self.top_k_tools != "all" and user_input is not None:
            tools = get_tools(
                query=user_input,
                llm=self.tools_llm,
                subagent_settings=self.subagents_settings,
                human=self.use_human_tool,
            )
        else:
            tools = make_all_tools(
                self.tools_llm,
                subagent_settings=self.subagents_settings,
                human=self.use_human_tool,
            )
        return AgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=AgentType.get_agent(self.agent_type).from_llm_and_tools(
                self.llm, tools
            ),
            handle_parsing_errors=True,
        )

    def run(self, user_input, callbacks=None):
        # todo: check this for both agent types

        self.agent = self._initialize_tools_and_agent(user_input)
        return self.agent.run(self.prompt.format(input=user_input), callbacks=callbacks)
