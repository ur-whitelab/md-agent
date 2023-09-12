import json
import os
import re
import select
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ...mainagent import _make_llm
from ...tools import PathRegistry
from ..prompts import ExplorePrompts, QAStep1Prompts, QAStep2Prompts, RefinePrompts


class ExplorerAgent:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-3.5-turbo",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
        confirm_on=True,
        mode="auto",
    ):
        load_dotenv()

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)
        self.llm_chain = self._initialize_llm()

        assert mode in ["auto", "manual"], f"mode {mode} not supported"
        self.mode = mode
        self.confirm_on = confirm_on

        self.ckpt_dir = ckpt_dir
        # self.path_registry = path_registry
        if not os.path.exists(f"{ckpt_dir}/curriculum/"):
            os.mkdir(f"{ckpt_dir}/curriculum/")

        # can remove this if we decide to use history from full_history instead
        if resume:
            print(f"Loading Curriculum Agent from {ckpt_dir}/curriculum")

            with open(f"{ckpt_dir}/curriculum/completed_tasks.json", "w") as f1:
                self.completed_tasks = json.load(f1)

            with open(f"{ckpt_dir}/curriculum/failed_tasks.json", "w") as f2:
                self.failed_tasks = json.load(f2)
        else:
            self.completed_tasks = []
            self.failed_tasks = []

    def _create_prompt(self):
        suffix = ""
        human_prompt = PromptTemplate(
            template=ExplorePrompts.PROMPT,
            input_variables=ExplorePrompts.INPUT_VARS,
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([ExplorePrompts.PREFIX, ExplorePrompts.FORMAT])
        )
        messages = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )
        return messages

    def _initialize_llm(self):
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(),
            callback_manager=StreamingStdOutCallbackHandler,
        )
        return llm_chain

    def run_llm(self, recent_history, full_history, skills, files):
        output = self.llm_chain(
            {
                "recent_history": {recent_history},
                "full_history": {full_history},
                "skills": {skills},
                "files": {files},
            }
        )
        return output


class RefiningCurriculumAgent:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-3.5-turbo",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
        confirm_on=True,
        mode="auto",
        qa_model="gpt-3.5-turbo",
    ):
        load_dotenv()

        # initialize agent
        llm = _make_llm(model, temp, verbose)
        qa_llm = _make_llm(qa_model, temp, verbose)
        self.llm_chain = self._initialize_llm(llm, RefinePrompts)
        self.qa_llm_step1 = self._initialize_llm(qa_llm, QAStep1Prompts)
        self.qa_llm_step2 = self._initialize_llm(qa_llm, QAStep2Prompts)

        assert mode in ["auto", "manual"], f"mode {mode} not supported"
        self.mode = mode
        self.confirm_on = confirm_on
        self.ckpt_dir = ckpt_dir
        os.makedirs(f"{ckpt_dir}/curriculum/", exist_ok=True)

        # can remove below if we decide to use history from full_history instead
        # it makes more sense if curriculum handles files of successes/failures thou
        if resume:
            with open(f"{ckpt_dir}/curriculum/completed_tasks.json", "w") as f1:
                self.completed_tasks = json.load(f1)

            with open(f"{ckpt_dir}/curriculum/failed_tasks.json", "w") as f2:
                self.failed_tasks = json.load(f2)
        else:
            self.completed_tasks = []
            self.failed_tasks = []

    def _create_prompt(self, prompts):
        suffix = ""
        human_prompt = PromptTemplate(
            template=prompts.PROMPT, input_variables=prompts.INPUT_VARS
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([prompts.PREFIX, prompts.FORMAT])
        )
        messages = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )
        return messages

    def _initialize_llm(self, llm, prompts):
        llm_chain = LLMChain(
            llm=llm,
            prompt=self._create_prompt(prompts),
        )
        return llm_chain

    def run_qa(self, info):
        questions = [
            f"What molecular dynamics tasks can I do with the files: {info['files']}?",
            # add other must-have questions here; do consider context length
        ]

        # Step 1: get questions
        q_response = self.qa_llm_step1(
            {
                "recent_history": info["recent_history"],
                "full_history": info["full_history"],
                "skills": info["skills"],
                "files": info["files"],
            }
        )
        try:
            pattern = r"Question \d+: (.*?[.?])"
            questions = re.findall(pattern, q_response)
        except Exception as e:
            return f"something went wrong. {e}"

        # Step 2: get answer for questions
        answers = []
        for question in questions:
            print(f"Curriculum Agent Question: {question}")
            answer = self.qa_llm_step2({"question": question})
            print(f"Curriculum Agent {answer}")
            answers.append(answer)

        # put together into a string
        qa_list = ""
        i = 1
        for question, answer in zip(questions, answers):
            if "Answer: Unknown" in answer or "language model" in answer:
                continue
            qa_list += f"Question {i}: {question}\n"
            qa_list += f"{answer}\n\n"
            i += 1
        return qa_list

    def propose_refined_task(self, task, original_prompt, qa_list, info):
        # ask curriculum agent to refine task (if action agent keeps failing)
        # manual mode is also available to manually enter task

        confirm_on = self.confirm_on
        mode = self.mode
        if mode == "manual":
            task = input("please enter the new task: ")
            assert task, ""
        elif mode == "auto":
            response = self.llm_chain(
                {
                    "task": task,
                    "original_task": original_prompt,
                    "qa_list": qa_list,
                    "recent_history": info["recent_history"],
                    "full_history": info["full_history"],
                    "skills": info["skills"],
                    "files": info["files"],
                }
            )
            # parse ai message
            try:
                task = ""
                for line in response.split("\n"):
                    if line.startswith("Task:"):
                        task = line[5:].replace(".", "").strip()
                assert task, "Task not found in Curriculum Agent response"
            except Exception as e:
                print(
                    f"""Error parsing refining curriculum response: {e}.
                Trying again!"""
                )
        else:
            raise ValueError(f"Invalid curriculum agent mode: {mode}")

        if task is None:
            return None
        else:
            if confirm_on:
                # have the user confirm the refined task by typing
                print(f"Task: {task}")
                print("Confirm? (y/n): ", end="", flush=True)
                timeout = 10  # sec
                inputs, _, _ = select.select([sys.stdin], [], [], timeout)
                if not inputs:
                    print(f"{timeout} seconds has passed. Proceeding with this task.")
                elif sys.stdin.readline().strip().lower() not in ["y", ""]:
                    print("Recommended task denied. Trying again.")
                    return None

                # alternative: no timeout
                # if input("Confirm? (y/n)").lower() not in ["y", ""]:
                # retries += 1
                # continue
            return task

    def run(self, task, original_prompt, info, max_retries=3):
        qa_list = self.run_qa(info)
        retries = 0
        while retries < max_retries:
            task = self.propose_refined_task(task, original_prompt, qa_list, info)
            if task:
                return task
            retries += 1
        raise RuntimeError("Max retries reached, failed to propose a task.")

    # def update_progress():

    # def clean_up_tasks():

    # def decompose_tasks():
