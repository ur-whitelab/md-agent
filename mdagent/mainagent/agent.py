import json
import os
import time

from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from mdagent.subagents import SubAgentSettings
from mdagent.utils import PathRegistry, SetCheckpoint, _make_llm

from ..tools import get_tools, make_all_tools
from .query_filter import make_prompt


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
        subagents_model="gpt-4-1106-preview",
        ckpt_dir="ckpt",
        resume=False,
        learn=True,
        top_k_tools=20,  # set "all" if you want to use all tools (& skills if resume)
        use_human_tool=False,
        curriculum=True,
        uploaded_files=[],  # user input files to add to path registry
    ):
        self.resume = resume
        self.ckpt_dir = ckpt_dir
        self.path_registry = PathRegistry.get_instance(
            resume=self.resume, ckpt_dir=self.ckpt_dir
        )
        self.ckpt_dir = self.path_registry.ckpt_dir
        self.uploaded_files = uploaded_files
        for file in uploaded_files:  # todo -> allow users to add descriptions?
            self.path_registry.map_path(file, file, description="User uploaded file")

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

        self.subagents_settings = SubAgentSettings(
            path_registry=self.path_registry,
            subagents_model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            verbose=verbose,
            resume=self.resume,
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
        self.prompt = make_prompt(user_input, self.agent_type, model="gpt-3.5-turbo")
        self.agent = self._initialize_tools_and_agent(user_input)
        return self.agent.run(self.prompt, callbacks=callbacks)

    # def run_and_eval(self, user_input, callbacks=None):
    #     self.agent = self._initialize_tools_and_agent(user_input)
    #     num_steps = 0
    #     tools_used = {}
    #     tools_details = {}
    #     step_start_time = start_time = time.time()
    #     for step in self.agent.iter({"input": user_input}, include_run_info=True):
    #         output = step.get("intermediate_step")
    #         if output:
    #             num_steps += 1
    #             action, observation = output[0]
    #             current_time = time.time()
    #             step_elapsed_time = current_time - step_start_time
    #             step_start_time = current_time
    #             tools_used[action.tool] = tools_used.get(action.tool, 0) + 1
    #             tools_details[f"Step {num_steps}"] = {
    #                 "tool": action.tool,
    #                 "tool_input": action.tool_input,
    #                 "observation": observation,
    #                 "step_elapsed_time (sec)": step_elapsed_time,
    #                 "timestamp_from_start (sec)": current_time - start_time,
    #             }
    #     final_output = step["output"]
    #     run_id = step["__run"].run_id
    #     total_seconds = time.time() - start_time
    #     total_mins = total_seconds / 60

    #     agent_settings = {
    #         "llm": self.llm.model_name,
    #         "agent_type": self.agent_type,
    #         "resume": self.subagents_settings.resume,
    #         "learn": not self.skip_subagents,
    #         "curriculum": self.subagents_settings.curriculum,
    #     }
    #     print("\n----- Evaluation Summary -----")
    #     print(f"Total Steps: {num_steps+1}")
    #     print(f"Total Time: {total_seconds:.2f} seconds ({total_mins:.2f} minutes)")

    #     summary = {
    #         "agent_settings": agent_settings,
    #         "total_steps": num_steps,
    #         "total_time_seconds": f"{total_seconds:.3f}",
    #         "total_time_minutes": f"{total_mins:.3f}",
    #         "final_answer": final_output,
    #         "tools_used": tools_used,
    #         "tools_details": tools_details,
    #         "run_id": str(run_id),
    #     }
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     os.makedirs(f"{self.ckpt_dir}/eval", exist_ok=True)
    #     filename = f"{self.ckpt_dir}/eval/evaluation_{timestamp}.json"
    #     with open(filename, "w") as f:
    #         json.dump(summary, f, indent=4)
    #     print(f"Summary saved to {filename}")
    #     return final_output

    def run_and_eval(self, user_input, callbacks=None, return_eval=False):
        self.agent = self._initialize_tools_and_agent(user_input)
        num_steps = 0
        tools_used = {}
        tools_details = {}
        failed_steps = 0
        step_start_time = start_time = time.time()
        last_step_status = ""
        second_last_step_status = ""

        for step in self.agent.iter({"input": user_input}, include_run_info=True):
            output = step.get("intermediate_step")
            if output:
                num_steps += 1
                action, observation = output[0]
                current_time = time.time()
                step_elapsed_time = current_time - step_start_time
                step_start_time = current_time
                tools_used[action.tool] = tools_used.get(action.tool, 0) + 1

                # Determine success or failure from the first sentence of the output
                first_sentence = observation.split(".")[
                    0
                ]  # Assuming sentences end with '.'
                if "Failed" in first_sentence or "Error" in first_sentence:
                    status = "Failed"
                elif "Succeeded" in first_sentence:
                    status = "Succeeded"
                else:
                    status = "Unclear"

                # Update step statuses
                second_last_step_status = last_step_status
                last_step_status = status

                tools_details[f"Step {num_steps}"] = {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": observation,
                    "status": status,  # Include success/failure status
                    "step_elapsed_time (sec)": f"{step_elapsed_time:.3f}",
                    "timestamp_from_start (sec)": f"{current_time - start_time:.3f}",
                }

        final_output = step.get("output", "")
        if "Succeeded" in final_output.split(".")[0]:
            prompt_passed = True
        elif "Failed" in final_output.split(".")[0]:
            prompt_passed = False
        else:
            # If the last step output doesn't explicitly state "Succeeded" or "Failed",
            # determine the success of the prompt based on the second last step status.
            prompt_passed = second_last_step_status != "Failed"

        run_id = step.get("__run", {}).get("run_id", "")
        total_seconds = time.time() - start_time
        total_mins = total_seconds / 60
        agent_settings = {
            "llm": self.llm.model_name,
            "agent_type": self.agent_type,
            "resume": self.subagents_settings.resume,
            "learn": not self.skip_subagents,
            "curriculum": self.subagents_settings.curriculum,
            "memory": self.subagents_settings.memory,
        }
        print("\n----- Evaluation Summary -----")
        print(f"Total Steps: {num_steps+1}")
        print(f"Total Time: {total_seconds:.2f} seconds ({total_mins:.2f} minutes)")

        summary = {
            "agent_settings": agent_settings,
            "prompt": user_input,
            "prompt_success": prompt_passed,
            "total_steps": num_steps,
            "failed_steps": failed_steps,
            "total_time_seconds": f"{total_seconds:.3f}",
            "total_time_minutes": f"{total_mins:.3f}",
            "final_answer": final_output,
            "tools_used": tools_used,
            "tools_details": tools_details,
            "run_id": str(run_id),
        }

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"{self.ckpt_dir}/eval", exist_ok=True)
        filename = f"{self.ckpt_dir}/eval/eval_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(summary, f, indent=4)

        if return_eval:
            brief_summary = {
                "agent_settings": agent_settings,
                "prompt_success": prompt_passed,
                "total_steps": num_steps,
                "failed_steps": failed_steps,
                "total_time_seconds": f"{total_seconds:.3f}",
                "run_id": str(run_id),
                "final_answer": final_output,
            }
            return final_output, brief_summary
        return final_output

    def force_clear_mem(self, all=False) -> str:
        if all:
            ckpt_dir = os.path.abspath(os.path.dirname(self.path_registry.ckpt_dir))
        else:
            ckpt_dir = self.path_registry.ckpt_dir
        confirmation = "nonsense"
        while confirmation.lower() not in ["yes", "no"]:
            confirmation = input(
                "Are you sure you want to"
                "clear memory? This will "
                "remove all saved "
                "checkpoints? (yes/no): "
            )

        if confirmation.lower() == "yes":
            set_ckpt = SetCheckpoint()
            set_ckpt.clear_all_ckpts(ckpt_dir)
            return "All checkpoints have been removed."
        else:
            return "Action canceled."
