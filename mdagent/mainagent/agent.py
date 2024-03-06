import json
import os
import time

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
        self.ckpt_dir = ckpt_dir
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

    def run(self, user_input, callbacks=None):
        self.agent = self._initialize_tools_and_agent(user_input)
        return self.agent.run(self.prompt.format(input=user_input), callbacks=callbacks)

    def run_and_eval(self, user_input, callbacks=None):
        self.agent = self._initialize_tools_and_agent(user_input)
        num_steps = 0
        tools_used = {}
        tools_details = {}
        step_start_time = start_time = time.time()
        for step in self.agent.iter({"input": user_input}, include_run_info=True):
            output = step.get("intermediate_step")
            if output:
                num_steps += 1
                action, observation = output[0]
                current_time = time.time()
                step_elapsed_time = current_time - step_start_time
                step_start_time = current_time
                tools_used[action.tool] = tools_used.get(action.tool, 0) + 1
                tools_details[f"Step {num_steps}"] = {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": observation,
                    "step_elapsed_time (sec)": step_elapsed_time,
                    "timestamp_from_start (sec)": current_time - start_time,
                }
        final_output = step["output"]
        run_id = step["__run"].run_id
        total_seconds = time.time() - start_time
        total_mins = total_seconds / 60

        agent_settings = {
            "llm": self.llm.model_name,
            "agent_type": self.agent_type,
            "resume": self.subagents_settings.resume,
            "learn": not self.skip_subagents,
            "curriculum": self.subagents_settings.curriculum,
        }
        print("\n----- Evaluation Summary -----")
        print(f"Total Steps: {num_steps+1}")
        print(f"Total Time: {total_seconds:.2f} seconds ({total_mins:.2f} minutes)")

        summary = {
            "agent_settings": agent_settings,
            "total_steps": num_steps,
            "total_time_seconds": f"{total_seconds:.3f}",
            "total_time_minutes": f"{total_mins:.3f}",
            "final_answer": final_output,
            "tools_used": tools_used,
            "tools_details": tools_details,
            "run_id": str(run_id),
        }
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"{self.ckpt_dir}/eval", exist_ok=True)
        filename = f"{self.ckpt_dir}/eval/evaluation_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to {filename}")
        return final_output
