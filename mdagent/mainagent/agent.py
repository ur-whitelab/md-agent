from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from mdagent.subagents import SubAgentSettings
from mdagent.tools import make_tools
from mdagent.utils import PathRegistry

load_dotenv()


main_prompt = PromptTemplate(
    inputs=["input"],
    template="""
    You are an expert molecular dynamics scientist and your
    task is to respond to the question or
    solve the problem to the best of your ability using
    the provided tools. Once you map a path to a short name,
    you may only use that short name in future actions.

    Here is the input:
    input: {input}
    """,
)


class MDAgent:
    def __init__(
        self,
        path_registry: Optional[PathRegistry] = None,
        tools=None,
        llm="gpt-4",
        temp=0.1,
        ckpt_dir="ckpt",
        resume=False,
    ):
        llm = ChatOpenAI(
            temperature=temp,
            model="gpt-4",
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        if path_registry is None:
            path_registry = PathRegistry.get_instance()

        self.subagents_settings = SubAgentSettings(
            path_registry=path_registry,
            subagents_model=llm,
            temp=temp,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )

        tools = make_tools(llm, self.subagents_settings)
        self.agent_instance = AgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=OpenAIFunctionsAgent.from_llm_and_tools(llm, tools),
            handle_parsing_errors=True,
        )

    def run(self, prompt):
        return self.agent_instance.run(main_prompt.format(input=prompt))
