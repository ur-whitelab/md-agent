from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from mdagent.tools import make_all_tools

load_dotenv()

# prompt template
main_prompt = PromptTemplate(
    inputs=["question"],
    template="""
    You are an expert molecular dynamics scientist
    and your task is to respond to the question or
    solve the problem to the best of your ability
    using the provided tools. Answer the question below using
    the most appropriate tools. If you do not have
    the necessary tools, you may make a tool to use
    later. Here is the question: {question}""",
)


class MDAgent:
    def __init__(
        self,
        tools=None,
        llm_name="gpt-4-1106-preview",
        temp=0.1,
        agent_type: str = "ZeroShotAgent",
    ):
        self.prompt = main_prompt
        llm = ChatOpenAI(
            temperature=temp,
            model=llm_name,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        tools = make_all_tools(llm)
        self.agent_instance = AgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=AgentType.get_agent(agent_type).from_llm_and_tools(llm, tools),
            handle_parsing_errors=True,
        )

    def run(self, question: str):
        return self.agent_instance.run(self.prompt.format(input=question))
