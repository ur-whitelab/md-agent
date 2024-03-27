import json
import os
import random
import string
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from mdagent.utils import PathRegistry

iterator_summary_template = PromptTemplate(
    input_variables=["history"],
    template="""
    Your job is to summarize the
    history of several attempts to
    solve a problem. Make sure to
    note the prompt, the mistakes
    made, and the final solution. You
    may include other details as
    well, if you think they are
    relevant.

    The history will be given as a
    json object with the following keys:

    - prompt: the prompt given to the solver
    - attempt_number: the number of the attempt
    - code: the code written by the solver
    - output: the output of the code
    - critique: the critique of the code
    - success: whether the attempt was successful

    Here is the history:
        {history}
        """,
)

agent_summary_template = PromptTemplate(
    input_variables=["agent_trace"],
    template="""
    Your job is to summarize the
    history of an agent's attempts to
    solve a problem. Be sure to
    include the prompt, every tool
    used (in order), and the final
    solution. You may include other
    details as well, if you think
    they are relevant.

    Here is the history:
        {agent_trace}
        """,
)


class MemoryManager:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-3.5-turbo",
        temp=0.1,
        ckpt_dir="ckpt",
        run_id="",
    ):
        load_dotenv()
        self.dir_name = f"{ckpt_dir}/memories"
        self.path_registry = path_registry
        self.run_id = run_id
        if not self.run_id:
            self.run_id = self.new_run_id()
            pull_mem = False
            self.run_id_mem = None
        else:
            pull_mem = True

        llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_iterator = LLMChain(llm=llm, prompt=iterator_summary_template)
        self.llm_agent_trace = LLMChain(llm=llm, prompt=agent_summary_template)

        os.makedirs(f"{self.dir_name}/memories", exist_ok=True)
        os.makedirs(f"{self.dir_name}/memories/memory_details", exist_ok=True)
        self.memory_path = (
            f"{self.dir_name}/memories/memory_details/run_{self.run_id}.json"
        )
        self.memory_summary_path = f"{self.dir_name}/memories/memory_summary.json"
        self.agent_summary_path = f"{self.dir_name}/memories/agent_summary.json"
        if pull_mem:
            self.run_id_mem = self.pull_agent_summary_from_mem(self.run_id)

    def new_run_id(self) -> str:
        length = 8
        characters = string.ascii_uppercase + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    def _write_to_json(self, data, path):
        if os.path.exists(path):
            data_original = json.load(open(path))
            data_original.update(data)
        else:
            data_original = data
        with open(path, "w") as f:
            f.write(json.dumps(data))

    def _write_history_iterator(
        self, prompt, attempt_number, code, output, critique, success=True
    ):
        data = {
            prompt
            + "_"
            + str(attempt_number): {
                "prompt": prompt,
                "attempt_number": attempt_number,
                "code": code,
                "output": output,
                "critique": critique,
                "success": success,
            }
        }
        self._write_to_json(data, self.memory_path)

    def _generate_summary_iterator(self):
        if not os.path.exists(self.memory_path):
            return None
        with open(self.memory_path, "r") as f:
            history = json.load(f)
        llm_out = self.llm_iterator({"history": json.dumps(history)})["text"]
        run_summary = {self.run_id: llm_out}
        self._write_to_json(run_summary, self.memory_summary_path)
        return None

    def pull_memory_summary(self, run_id: str = ""):
        if not run_id:
            run_id = self.run_id
        if not os.path.exists(self.memory_summary_path):
            return None
        with open(self.memory_summary_path, "r") as f:
            summary = json.load(f)
        if run_id not in summary:
            return None
        return summary[run_id]

    def generate_summary(self, agent_trace):
        llm_out = self.llm_agent_trace({"agent_trace": agent_trace})["text"]
        run_summary = {self.run_id: llm_out}
        self._write_to_json(run_summary, self.agent_summary_path)
        return None

    def retrieve_recent_memory_iterator(self, last_only=False):
        if not os.path.exists(self.memory_path):
            return str({})
        with open(self.memory_path, "r") as f:
            memories = json.load(f)
        if last_only:
            return str(memories[list(memories.keys())[-1]])
        return json.dumps(memories)

    def write_all_summaries(self, agent_trace):
        self.generate_summary(agent_trace)
        self._generate_summary_iterator()

    def pull_agent_summary_from_mem(self, run_id: str = ""):
        if not run_id:
            run_id = self.run_id
        if not os.path.exists(self.agent_summary_path):
            return None
        with open(self.agent_summary_path, "r") as f:
            summary = json.load(f)
        if run_id not in summary:
            return None
        return summary[run_id]
