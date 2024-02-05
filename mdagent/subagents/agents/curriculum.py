import json
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from mdagent.utils import PathRegistry

from .prompts import curriculum_template, curriculum_template_next_step

load_dotenv()


class Curriculum:
    def __init__(
        self,
        model="gpt-4-1106-preview",
        temp=0.1,
        path_registry: Optional[PathRegistry] = None,
    ):
        self.llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.path_registry = path_registry

    def run_plan(self, task, curr_tools, files, failed_tasks=""):
        llm_chain = LLMChain(llm=self.llm, prompt=curriculum_template)
        message = llm_chain(
            {
                "final_task": task,
                "tools": curr_tools,
                "files": files,
                "failed_tasks": failed_tasks,
            }
        )["text"]

        if message.startswith("```json"):
            # Remove the triple backticks and 'json'
            message = message.strip("`").replace("json\n", "")

        parsed_message = json.loads(message)
        rationale = parsed_message.get("Rationale", "")
        subtasks = parsed_message.get("Plan", [])
        return rationale, subtasks

    def run_next(self, info):
        llm_chain = LLMChain(llm=self.llm, prompt=curriculum_template_next_step)
        message = llm_chain(
            {
                "user_prompt": info["user_prompt"],
                "current_stage": info["current_stage"],
                "exploration_request": info["explore"],
                "available_files": info["files"],
                "tools_access": info["current_tools"],
                "all_tools": info["all_tools"],
                "failed_task": info["failed_task"],
                "failed_tool": info["failed_tool"],
                "tool_inputs": info["tool_inputs"],
                "tool_output": info["tool_output"],
                "succeeded_tasks": info["succeeded_tasks"],
            }
        )["text"]

        if message.startswith("```json"):
            # Remove the triple backticks and 'json'
            message = message.strip("`").replace("json\n", "")

        parsed_message = json.loads(message)
        rationale = parsed_message.get("Reasoning", "")
        task = parsed_message.get("Task", "")
        return rationale, task
