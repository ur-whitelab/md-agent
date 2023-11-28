from typing import Optional

from dotenv import load_dotenv
from langchain.chains import LLMChain

from mdagent.subagents.prompts import curriculum_template
from mdagent.utils import PathRegistry, _make_llm


class CurriculumAgent:
    def __init__(
        self,
        model="gpt-4-1106-preview",
        temp=0.1,
        verbose=True,
        path_registry: Optional[PathRegistry] = None,
    ):
        load_dotenv()

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)
        self.llm_chain = LLMChain(llm=self.llm, prompt=curriculum_template)
        self.path_registry = path_registry

    def run(self, task, curr_tools, files, failed_tasks=""):
        message = self.llm_chain(
            {
                "final_task": task,
                "tools": curr_tools,
                "files": files,
                "failed_tasks": failed_tasks,
            }
        )["text"]
        # TODO: parse a list of subtasks from the message
        return message
