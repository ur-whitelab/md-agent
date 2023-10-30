from dotenv import load_dotenv
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from mdagent.mainagent.prompt import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, SUFFIX
from mdagent.subagents import SubAgentSettings
from mdagent.tools import get_tools
from mdagent.utils import PathRegistry, _make_llm

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
        path_registry=None,
        subagents_model="gpt-3.5-turbo",
        ckpt_dir="ckpt",
        resume=False,
        top_k_tools=10,
    ):
        self.llm = _make_llm(model, temp, verbose)
        self.tools_llm = _make_llm(tools_model, temp, verbose)
        self.tools = tools
        self.top_k_tools = top_k_tools
        self.ckpt_dir = ckpt_dir
        if path_registry is None:
            path_registry = PathRegistry.get_instance()
        self.subagents_settings = SubAgentSettings(
            path_registry=path_registry,
            subagents_model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )

    def run(self, prompt):
        # get necessary tools based on prompt
        if self.tools is None:
            tools = get_tools(
                query=prompt,
                llm=self.tools_llm,
                subagent_settings=self.subagents_settings,
                ckpt_dir=self.ckpt_dir,
                retrieval_top_k=self.top_k_tools,
            )
        else:
            tools = self.tools

        # initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools=tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=self.subagents_settings.max_iterations,
            return_intermediate_steps=True,
        )

        # run the agent
        outputs = self.agent_executor({"input": prompt})
        # Parse long output (with intermediate steps)
        intermed = outputs["intermediate_steps"]
        final = ""
        for step in intermed:
            final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
        final += f"Final Answer: {outputs['output']}"

        return final
