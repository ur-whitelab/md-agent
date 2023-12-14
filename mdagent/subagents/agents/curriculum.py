import json
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from mdagent.utils import PathRegistry

from .prompts import curriculum_template

load_dotenv()


class Curriculum:
    def __init__(
        self,
        model="gpt-4",
        temp=0.1,
        path_registry: Optional[PathRegistry] = None,
    ):
        llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_chain = LLMChain(llm=llm, prompt=curriculum_template)
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

        if message.startswith("```json"):
            # Remove the triple backticks and 'json'
            message = message.strip("`").replace("json\n", "")

        parsed_message = json.loads(message)
        rationale = parsed_message.get("Rationale", "")
        subtasks = parsed_message.get("Plan", [])
        return rationale, subtasks
