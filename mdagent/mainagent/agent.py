import json

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from mdagent.subagents import SubAgentSettings
from mdagent.utils import PathRegistry, _make_llm

from ..tools import get_tools, make_all_tools
from .prompt import modular_analysis_prompt, openaifxn_prompt, structured_prompt
from .query_filter import Parameters, Task_type, create_filtered_query

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
        agent_type="OpenAIFunctionsAgent",  # this can also be structured_chat
        model="gpt-4-1106-preview",  # current name for gpt-4 turbo
        tools_model="gpt-4-1106-preview",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        path_registry=None,
        subagents_model="gpt-4-1106-preview",
        ckpt_dir="ckpt",
        resume=False,
        learn=True,
        top_k_tools=20,  # set "all" if you want to use all tools (& skills if resume)
        use_human_tool=False,
        curriculum=True,
        uploaded_files=[],  # user input files to add to path registry
    ):
        if path_registry is None:
            path_registry = PathRegistry.get_instance()
        self.uploaded_files = uploaded_files
        for file in uploaded_files:  # todo -> allow users to add descriptions?
            path_registry.map_path(file, file, description="User uploaded file")

        self.agent_type = agent_type
        self.user_tools = tools
        self.tools_llm = _make_llm(tools_model, temp, verbose)
        self.top_k_tools = top_k_tools
        self.use_human_tool = use_human_tool

        self.llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        if learn:
            self.skip_subagents = False
        else:
            self.skip_subagents = True

        # PR Comment: moved the initialization of the prompt (as it will now depend
        # on the agent_type and user input) inside the run method

        self.subagents_settings = SubAgentSettings(
            path_registry=path_registry,
            subagents_model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            verbose=verbose,
            ckpt_dir=ckpt_dir,
            resume=resume,
            curriculum=curriculum,
        )

    def _initialize_tools_and_agent(self, user_input=None):
        """Retrieve tools and initialize the agent."""
        if self.user_tools is not None:
            self.tools = self.user_tools
        else:
            if self.top_k_tools != "all" and user_input is not None:
                # retrieve only tools relevant to user input
                self.tools = get_tools(
                    query=user_input,
                    llm=self.tools_llm,
                    subagent_settings=self.subagents_settings,
                    human=self.use_human_tool,
                    skip_subagents=self.skip_subagents,
                )
            else:
                # retrieve all tools, including new tools if any
                self.tools = make_all_tools(
                    self.tools_llm,
                    subagent_settings=self.subagents_settings,
                    human=self.use_human_tool,
                    skip_subagents=self.skip_subagents,
                )
        return AgentExecutor.from_agent_and_tools(
            tools=self.tools,
            agent=AgentType.get_agent(self.agent_type).from_llm_and_tools(
                self.llm,
                self.tools,
            ),
            handle_parsing_errors=True,
        )

    # PR Comment: The run method is now responsible for initializing the prompt too!
    # If you're reviewing this, strongly recommend to look at the query_filter.py file
    # first
    def run(self, user_input, callbacks=None):
        if self.agent_type == "Structured":
            tries = 1  # PR Comment: trying 3 times (robustness) before defaulting
            # to the un processed input with the previous prompt

            while tries <= 3:
                try:
                    structured_query = create_filtered_query(
                        user_input, model="gpt-3.5-turbo"
                    )
                    structured_query = json.loads(structured_query)
                    parameters = Parameters.parse_parameters_string(
                        structured_query["Parameters"]
                    )
                    _parameters = ""
                    for key, value in parameters.items():
                        if value == "None":
                            continue
                        else:
                            _parameters += f"{key}: {value}, "
                    _plan = ""
                    if structured_query["UserProposedPlan"] == "[]":
                        _plan += "None"
                    else:
                        if type(structured_query["UserProposedPlan"]) == str:
                            for plan in structured_query["UserProposedPlan"].split(","):
                                _plan += f"{plan},"
                        elif type(structured_query["UserProposedPlan"]) == list:
                            for plan in structured_query["UserProposedPlan"]:
                                _plan += f"{plan},"
                    _proteins = ""
                    if structured_query["ProteinS"] == "['None']":
                        _proteins += "None"
                    elif structured_query["ProteinS"] == "[]":
                        _proteins += "None"
                    else:
                        for protein in eval(structured_query["ProteinS"]):
                            _proteins += f"{protein}, "
                    _subtasks = ""
                    if structured_query["Subtask_types"] == "['None']":
                        _subtasks += "None"
                    elif structured_query["Subtask_types"] == "[]":
                        _subtasks += "None"
                    elif structured_query["Subtask_types"] == ["None"]:
                        _subtasks += "None"
                    else:
                        if type(structured_query["Subtask_types"]) == str:
                            for subtask in Task_type.parse_task_type_string(
                                structured_query["Subtask_types"]
                            ):
                                _subtasks += f"{subtask}, "
                        elif type(structured_query["Subtask_types"]) == list:
                            for subtask in structured_query["Subtask_types"]:
                                _str = Task_type.parse_task_type_string(subtask)
                                _subtasks += f"{_str}, "  # PR Comment: Two steps
                                # to stay within char limit
                    prompt = modular_analysis_prompt.format(
                        Main_Task=structured_query["Main_Task"],
                        Subtask_types=_subtasks,
                        Proteins=_proteins,
                        Parameters=_parameters,
                        UserProposedPlan=_plan,
                    )
                    break
                except ValueError as e:
                    print(f"Failed to structure query, attempt {tries}/3. Retrying...")
                    print(e, e.args)
                    tries += 1
                    continue
                except Exception as e:
                    print(f"Failed to structure query, attempt {tries}/3. Retrying...")
                    print(e, e.args)
                    tries += 1
                    continue

            # PR Comment: Assigning the prompt attribute
            if tries > 3:
                # PR Comment: In case the structured query fails after 3 attempts,
                # the input will be used as we've been doing it.
                print(
                    "Failed to structure query after 3 attempts."
                    "Input will be used as is."
                )
                self.prompt = structured_prompt.format(input=user_input)
            else:
                # PR Comment: If the structured query is successful, the prompt will
                # be set
                self.prompt = prompt
        elif self.agent_type == "OpenAIFunctionsAgent":
            self.prompt = openaifxn_prompt.format(input=user_input)

        self.agent = self._initialize_tools_and_agent(user_input)
        # PR Comment: The prompt attribute is already set
        return self.agent.run(self.prompt, callbacks=callbacks)
