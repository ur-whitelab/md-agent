import json
import os
import random
import string

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
    include the prompt, every step,
    and the final
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
        path_registry: PathRegistry,
        model="gpt-3.5-turbo",
        temp=0.1,
        run_id="",
    ):
        self.path_registry = path_registry
        self.dir_name = f"{path_registry.ckpt_memory}"
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

        self._make_all_dirs()
        if pull_mem:
            self.run_id_mem = self.pull_agent_summary_from_mem(self.run_id)

    def _make_all_dirs(self):
        """
        This function creates the directories for the memory manager.

        Parameters:
        - None

        Returns:
        - None
        """
        self.cnt_history_dir = f"{self.dir_name}/cnt_history"
        self.cnt_history_details_dir = f"{self.dir_name}/cnt_history/details"

        os.makedirs(self.cnt_history_dir, exist_ok=True)
        os.makedirs(self.cnt_history_details_dir, exist_ok=True)

        self.cnt_history_details = (
            f"{self.cnt_history_details_dir}/run_{self.run_id}.json"
        )
        self.agent_trace_summary = f"{self.dir_name}/agent_run_summaries.json"

    def new_run_id(self) -> str:
        """
        This function generates a new run_id.

        Parameters:
        - None

        Returns:
        - str: The new run_id.
        """
        length = 8
        characters = string.ascii_uppercase + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    def _write_to_json(self, data, path):
        """
        This function writes data to a json file.

        Parameters:
        - data (dict): The data to write to the json file.
        - path (str): The path to the json file.

        Returns:
        - None
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                data_original = json.load(f)
            data_original.update(data)
        else:
            data_original = data
        with open(path, "w") as f:
            f.write(json.dumps(data_original))

    ### Iterator/CreateNewTool Functions ###

    # not currently used
    def _generate_summary_iterator(self, history: dict = {}):
        """ "
        This function generates a summary of the iterator history.

        Parameters:
        - history (dict): The history of the iterator.

        Returns:
        - str: The summary of the iterator history.
        """
        if not history:
            # get last history
            history = self.retrieve_recent_memory_iterator(last_only=True)
        llm_out = self.llm_iterator({"history": json.dumps(history)})["text"]
        return llm_out

    def _write_history_iterator(
        self,
        prompt,
        code,
        output,
        critique,
        success=True,
        summary=False,
        new_iteration=False,
    ):
        """
        This function writes the iteration (create new tool)
        history to a json file.
        These files will live within the cnt_history/details
        directory of the
        current ckpt_dir and will be identified by the run_id.

        Parameters:
        - prompt (str): The prompt given to the solver.
        - code (str): The code written by the solver.
        - output (str): The output of the code.
        - critique (str): The critique of the code.
        - success (bool): A flag indicating whether the
        attempt was successful. Default is True.
        - summary (bool): A flag indicating whether to
        generate a summary of the history. Default is False.
        - new_iteration (bool): A flag indicating whether
        to start a new iteration instance (True)
        or to continue with the current iteration instance
        by incrementing the attempt count (False). Default is False.

        Returns:
        - None"""

        history = {
            "prompt": prompt,
            "code": code,
            "output": output,
            "critique": critique,
            "success": success,
        }
        if summary:  # not currently used
            llm_summary = self._generate_summary_iterator(history)
            history["summary"] = llm_summary
        else:
            history["summary"] = None

        iter_num = self.get_iteration_number(new_iteration=new_iteration)
        data = {iter_num: history}
        self._write_to_json(data, self.cnt_history_details)

    def get_iteration_number(self, new_iteration=False):
        """
        Retrieves the next number for iteration.

        Parameters:
        - new_iteration (bool): A flag indicating whether
        to start a new iteration instance (True)
        or to continue with the current iteration instance
        by incrementing the attempt count (False). Default is False.

        Returns:
        - str: The next iteration number in the format "instance.attempt".
        If no previous history exists or the history file
        is empty, it returns "0.0".
        """

        if not os.path.exists(self.cnt_history_details):
            return str(0.0)
        with open(self.cnt_history_details, "r") as f:
            details = json.load(f)
        if not details:
            return str(0.0)
        keys = list(details.keys())
        instance, attempt = keys[-1].split(".")
        if new_iteration:
            instance = str(int(instance) + 1)
            attempt = "0"
            return f"{instance}.{attempt}"
        else:
            attempt = str(int(attempt) + 1)
            return f"{instance}.{attempt}"

    def retrieve_iterator_details(self, run_id: str = "", iter_num: str = ""):
        """
        This function pulls the iterator details for a given
        run_id from the cnt_history/details directory of the
        current ckpt_dir.

        Parameters:
        - run_id (str): The run_id to pull the details for.
        Default is "".
        - iter_num (str): The iteration number to pull the
        details for. Default is "".

        Returns:
        - dict: The iterator details for the given run_id and
        iteration number.
        If no previous history exists or the history file is
        empty, it returns None.
        """
        if not run_id:
            run_id = self.run_id
        if not os.path.exists(self.cnt_history_details):
            return None
        with open(self.cnt_history_details, "r") as f:
            details = json.load(f)
        if iter_num:
            details = {k: v for k, v in details.items() if f"{iter_num}." in k}
        return details

    def retrieve_recent_memory_iterator(self, last_only=False):
        """
        This function pulls the most recent memory from the
        memory file.

        Parameters:
        - last_only (bool): A flag indicating whether to pull
        the most recent memory only (True)
        or to pull all memories (False). Default is False.

        Returns:
        - str: The most recent memory in the format "memory".
        If no previous history exists or the history file is
        empty, it returns "{}".
        """
        if not os.path.exists(self.cnt_history_details):
            return str({})
        with open(self.cnt_history_details, "r") as f:
            memories = json.load(f)
        if last_only:
            return str(memories[list(memories.keys())[-1]])
        return json.dumps(memories)

    ### Agent/Run Summary Functions ###

    def get_summary_number(self):
        """
        Retrieves the next number for the agent summary.

        Parameters:
        - None

        Returns:
        - str: The next summary number.
        If no previous history exists or the history file is empty,
        it returns "0".
        """
        if not os.path.exists(self.agent_trace_summary):
            return str(0)
        with open(self.agent_trace_summary, "r") as f:
            summary = json.load(f)
        if not summary:
            return str(0)
        return str(len(summary.keys()))

    def generate_agent_summary(self, agent_trace):
        """
        This function generates a summary of the agent's run history
        and writes it to a json file.
        There will be one file per run_id.

        Parameters:
        - agent_trace (dict): The agent's run history.

        Returns:
        - None
        """
        llm_out = self.llm_agent_trace({"agent_trace": agent_trace})["text"]
        key_str = f"{self.run_id}.{self.get_summary_number()}"
        run_summary = {key_str: llm_out}
        self._write_to_json(run_summary, self.agent_trace_summary)
        return None

    def pull_agent_summary_from_mem(self, run_id: str, run_num: int = 0):
        """
        This function pulls the agent summary for a given run
        number from the current run_id's agent summary file.

        Parameters:
        - run_num (int): The run number to pull the summary for.
        Default is 0, which pulls the most recent summary.

        Returns:
        - str: The agent summary for the given run number.
        If the summary does not exist, it returns None.
        """
        run_id_full = f"{run_id}.{run_num}"
        if not os.path.exists(self.agent_trace_summary):
            return "Path does not exist."
        with open(self.agent_trace_summary, "r") as f:
            summary = json.load(f)
        if run_id_full not in summary.keys():
            return "Run ID not found. Keys are: " + str(summary.keys())
        return summary[run_id_full]
