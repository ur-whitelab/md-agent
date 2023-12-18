from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools.render import format_tool_to_openai_function

from mdagent.subagents import SubAgentSettings
from mdagent.utils import PathRegistry, _make_llm

from ..tools import make_all_tools
from .prompt import (
    FORMAT_INSTRUCTIONS,
    FORMAT_INSTRUCTIONS_FUNC,
    PREFIX,
    QUESTION_PROMPT,
    QUESTION_PROMPT_FUNC,
    SUFFIX,
)

load_dotenv()


main_prompt = PromptTemplate(
    input_variables=["input"],
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
        tools=None,
        model="gpt-4-1106-preview",  # current name for gpt-4 turbo
        tools_model="gpt-4-1106-preview",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
        path_registry=None,
        subagents_model="gpt-4-1106-preview",
        ckpt_dir="ckpt",
        resume=False,
        top_k_tools=10,
        use_human_tool=False,
        type="functions",
    ):
        self.type = type
        if tools is None:
            tools_llm = _make_llm(tools_model, temp, verbose)
            tools = make_all_tools(tools_llm)

        if self.type == "functions":
            self.llm = _make_llm("gpt-4-1106-preview", temp, verbose)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "\n\n".join([PREFIX, FORMAT_INSTRUCTIONS_FUNC])),
                    ("user", QUESTION_PROMPT_FUNC),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            llm_with_tools = self.llm.bind(
                functions=[format_tool_to_openai_function(t) for t in tools]
            )
            agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_function_messages(
                        x["intermediate_steps"]
                    ),
                }
                | prompt
                | llm_with_tools
                | OpenAIFunctionsAgentOutputParser()
            )

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                handle_parsing_errors=True,
                max_iterations=40,
                return_intermediate_steps=True,
                verbose=True,
            )
        elif type == "structured" or type == "mrkl":
            # Initialize agent
            self.llm = _make_llm(model, temp, verbose)
            self.agent_executor = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
                return_intermediate_steps=True,
                max_iterations=max_iterations,
                handle_parsing_errors=True,
            )

        self.llm = _make_llm(model, temp, verbose)
        self.tools_llm = _make_llm(tools_model, temp, verbose)
        self.tools = tools
        self.top_k_tools = top_k_tools
        self.ckpt_dir = ckpt_dir
        self.human = use_human_tool
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
            retrieval_top_k=top_k_tools,
        )

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        # Parse long output (with intermediate steps)
        intermed = outputs["intermediate_steps"]

        final = ""
        if self.type == "mrkl":
            for step in intermed:
                final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
            final += f"Final Answer: {outputs['output']}"

            return final
        if self.type == "functions":
            for step in intermed:
                final += f"{step[0].log}\n" + f"Observation: {step[1]}\n"
            final += f"Final Answer: {outputs['output']}"

            return final
