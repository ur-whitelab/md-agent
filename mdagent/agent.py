from dotenv import load_dotenv
from langchain import agents
from rmrkl import ChatZeroShotAgent

from . import make_llm
from .agent_prompt import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, SUFFIX

load_dotenv()


class MDAgent:
    def __init__(
        self,
        tools=None,
        model="gpt-4",
        tools_model="gpt-4",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
    ):
        self.llm = make_llm(model, temp, verbose)

    def make_tools(self):
        tools_llm = make_llm(self.tools_model, self.temp, self.verbose)
        all_tools = agents.load_tools(["python_repl", "human", "llm-math"], tools_llm)
        # add in tools from tool library
        return all_tools

        # Initialize agent

    def init_agent(self, tools):
        self.agent = ChatZeroShotAgent.from_llm_and_tools(
            self.llm,
            tools=tools,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            question_prompt=QUESTION_PROMPT,
        )
        return None

    def run(self, prompt):
        # get tools
        status = False
        tools = self.make_tools()
        self.init_agent(tools)
        try:
            outputs = self.agent({"input": prompt})
            # Parse long output (with intermediate steps)
            intermed = outputs["intermediate_steps"]

            final = ""
            for step in intermed:
                final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
            final += f"Final Answer: {outputs['output']}"

            return status, final
        except Exception as e:
            return status, str(e)
