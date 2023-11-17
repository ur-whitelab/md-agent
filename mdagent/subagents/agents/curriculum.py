import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from mdagent.utils import PathRegistry

from .prompts import (
    explore_prompt_template,
    qa1_prompt_template,
    qa2_prompt_template,
    refine_prompt_template,
)


class Explorer:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-3.5-turbo",
        temp=0.1,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
        confirm_on=True,
        mode="auto",
    ):
        load_dotenv()

        # initialize agent
        self.llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=explore_prompt_template)
        assert mode in ["auto", "manual"], f"mode {mode} not supported"
        self.mode = mode
        self.confirm_on = confirm_on
        self.path_registry = path_registry
        self.ckpt_dir = ckpt_dir
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

    def run_llm(self, full_history, skills, files):
        output = self.llm_chain(
            {
                "full_history": {full_history},
                "skills": {skills},
                "files": {files},
            }
        )
        return output


class RefiningCurriculum:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        temp=0.1,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
        confirm_on=True,
        mode="auto",
        qa_model="gpt-3.5-turbo",
    ):
        load_dotenv()

        self.llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=refine_prompt_template)
        self.qa_llm_step1 = LLMChain(llm=self.llm, prompt=qa1_prompt_template)
        self.qa_llm_step2 = LLMChain(llm=self.llm, prompt=qa2_prompt_template)

        assert mode in ["auto", "manual"], f"mode {mode} not supported"
        self.mode = mode
        self.confirm_on = confirm_on
        self.ckpt_dir = ckpt_dir
        os.makedirs(f"{ckpt_dir}/curriculum/", exist_ok=True)

        # TODO: clean up/manage history files between action, curriculum, and critic
        if resume:
            with open(f"{ckpt_dir}/curriculum/completed_tasks.json", "w") as f1:
                self.completed_tasks = json.load(f1)

            with open(f"{ckpt_dir}/curriculum/failed_tasks.json", "w") as f2:
                self.failed_tasks = json.load(f2)
        else:
            self.completed_tasks = []
            self.failed_tasks = []

    def _run_qa(self, info):
        # Step 1: get questions
        questions = []
        if info["files"]:  # if not empty
            questions += [
                "What molecular dynamics tasks can I do with the following"
                f" files: {info['files']}?"
            ]
        if info["skills"]:
            questions += [
                "What molecular dynamics tasks can I do with the following"
                f" skills: {info['skills']}?"
            ]
        q_response = self.qa_llm_step1(
            {
                "full_history": info["full_history"],
                "skills": info["skills"],
                "files": info["files"],
            }
        )["text"]
        try:
            pattern = r"Question \d+: (.*?[.?])"
            questions.extend(re.findall(pattern, q_response))
        except Exception as e:
            return f"something went wrong. {e}"

        # Step 2: get answer for questions
        answers = []
        for question in questions:
            print(f"\n\nQuestion: {question}")
            answer = self.qa_llm_step2({"question": question})["text"]
            if not self.verbose:
                print(f"\n{answer}")
            answers.append(answer)

        # put together into a string
        qa_list = ""
        i = 1
        for question, answer in zip(questions, answers):
            if "Answer: Unknown" in answer or "As an AI assistant" in answer:
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
                    "full_history": info["full_history"],
                    "skills": info["skills"],
                    "files": info["files"],
                }
            )["text"]
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
                print(f"\n\033[1;34mTask: {task}\033[00m")
                if input("Confirm? (y/n)").lower() not in ["y", ""]:
                    return None

                # alternative: with timeout ---> currently doesn't work
                # print("Confirm? (y/n): ", end="", flush=True)
                # timeout = 10  # sec
                # inputs, _, _ = select.select([sys.stdin], [], [], timeout)
                # if not inputs:
                #     print(f"{timeout} seconds has passed. Proceeding with this task.")
                # elif sys.stdin.readline().strip().lower() not in ["y", ""]:
                #     print("Recommended task denied. Trying again.")
                #     return None
            return task

    def run(self, task, original_prompt, info, max_retries=3):
        qa_list = self._run_qa(info)
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
