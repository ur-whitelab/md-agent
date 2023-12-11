import json
from json import JSONDecodeError
from typing import List, Union

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.agent import AgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    ChatGeneration,
    Generation,
    OutputParserException,
)
from langchain.schema.agent import AgentActionMessageLog
from langchain.tools.render import format_tool_to_openai_function

from mdagent.subagents import SubAgentSettings
from mdagent.utils import PathRegistry, _make_llm

from ..tools import make_all_tools
from .prompt import (
    FINAL_ANSWER_ACTION,
    FORMAT_INSTRUCTIONS,
    FORMAT_INSTRUCTIONS_FUNC,
    PREFIX,
    QUESTION_PROMPT,
    QUESTION_PROMPT_FUNC,
    SUFFIX,
)

load_dotenv()


class customOpenAIFunctionsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent action/finish.

    Is meant to be used with OpenAI models, as it relies on the specific
    function_call parameter from OpenAI to convey what tools to use.

    If a function_call parameter is passed, then that is used to get
    the tool and tool input.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "openai-functions-agent"

    @staticmethod
    def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse an AI message."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        function_call = message.additional_kwargs.get("function_call", {})

        if function_call:
            function_name = function_call["name"]
            try:
                _tool_input = json.loads(function_call["arguments"])
            except JSONDecodeError:
                raise OutputParserException(
                    f"Could not parse tool input: {function_call} because "
                    f"the `arguments` is not valid JSON."
                )

            # HACK HACK HACK:
            # The code that encodes tool input into Open AI uses a special variable
            # name called `__arg1` to handle old style tools that do not expose a
            # schema and expect a single string argument as an input.
            # We unpack the argument here if it exists.
            # Open AI does not support passing in a JSON array as an argument.
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            return AgentActionMessageLog(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            )
        if FINAL_ANSWER_ACTION in message.content:
            return AgentFinish(
                {"output": message.content.split(FINAL_ANSWER_ACTION)[-1].strip()},
                log=message.content,
            )
        raise (OutputParserException(f"Could not parse message: {message}"))

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise ValueError("Can only parse messages")


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
            tools = make_all_tools(tools_llm, verbose=verbose)

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
                | customOpenAIFunctionsAgentOutputParser()
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
