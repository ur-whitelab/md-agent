from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent

from mdagent.agent.prompt import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, SUFFIX
from mdagent.tools import make_all_tools
from mdagent.utils import _make_llm

load_dotenv()


class MDAgent:
    def __init__(
        self,
        tools=None,
        model="gpt-4-1106-preview",  # current name for gpt-4 turbo
        tools_model="gpt-4-1106-preview",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
    ):
        self.llm = _make_llm(model, temp, verbose)
        if tools is None:
            tools_llm = _make_llm(tools_model, temp, verbose)
            tools = make_all_tools(tools_llm, verbose=verbose)

        # Initialize agent
        # self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
        #    tools=tools,
        # agent=ChatZeroShotAgent.from_llm_and_tools(
        #    self.llm,
        #    tools=tools,
        #    suffix=SUFFIX,
        #    format_instructions=FORMAT_INSTRUCTIONS,
        #    question_prompt=QUESTION_PROMPT,
        # ),
        #    verbose=True,
        #    max_iterations=max_iterations,
        #    return_intermediate_steps=True,
        # )
        self.agent_executor = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            question_prompt=QUESTION_PROMPT,
            return_intermediate_steps=True,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        # Parse long output (with intermediate steps)
        # intermed = outputs["intermediate_steps"]

        # final = ""
        # for step in intermed:
        #     final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
        final = outputs["output"]

        return final
