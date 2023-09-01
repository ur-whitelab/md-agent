import json
import os

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

from ...mdagent.agent import _make_llm
from ...prompts import ExplorePrompts, QAStep1Prompts, QAStep2Prompts, RefinePrompts


class Explorer:
    def __init__(
        self,
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
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(),
            callback_manager=StreamingStdOutCallbackHandler,
        )
        assert mode in ["auto", "manual"], f"mode {mode} not supported"
        self.mode = mode
        self.confirm_on = confirm_on

        self.ckpt_dir = ckpt_dir
        if not os.path.exists(f"{ckpt_dir}/curriculum/"):
            os.mkdir(f"{ckpt_dir}/curriculum/")

        # can remove this if we decide to use history from full_history instead
        if resume:
            print(f"\033[35mLoading Curriculum Agent from {ckpt_dir}/curriculum\033[0m")

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
            input_variables=[
                "recent_history",
                "full_history",
                "skills",
                "files",
            ],
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


class QAStep1Agent:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        load_dotenv()

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(),
            callback_manager=StreamingStdOutCallbackHandler,
        )

    def _create_prompt(self):
        suffix = ""
        human_prompt = PromptTemplate(
            template=QAStep1Prompts.PROMPT,
            input_variables=[
                "recent_history",
                "full_history",
                "skills",
                "files",
            ],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([QAStep1Prompts.PREFIX, QAStep1Prompts.FORMAT])
        )
        messages = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )
        return messages

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


class QAStep2Agent:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        load_dotenv()

        # initialize agent
        self.llm = _make_llm(model, temp, verbose)
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(),
            callback_manager=StreamingStdOutCallbackHandler,
        )

    def _create_prompt(self):
        suffix = ""
        human_prompt = PromptTemplate(
            template=QAStep2Prompts.PROMPT, input_variables=["question"]
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([QAStep2Prompts.PREFIX, QAStep2Prompts.FORMAT])
        )
        messages = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )
        return messages

    def run_llm(self, question):
        return self.llm_chain({"question": {question}})


class RefiningCurriculum:
    def __init__(
        self,
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
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._create_prompt(),
            callback_manager=StreamingStdOutCallbackHandler,
        )
        assert mode in ["auto", "manual"], f"mode {mode} not supported"
        self.mode = mode
        self.confirm_on = confirm_on
        self.ckpt_dir = ckpt_dir
        os.makedirs(f"{ckpt_dir}/curriculum/", exist_ok=True)

        # can remove below if we decide to use history from full_history instead
        # it makes more sense if curriculum handles files of successes/failures tho
        if resume:
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
            template=RefinePrompts.PROMPT,
            input_variables=[
                "qa_list",
                "recent_history",
                "full_history",
                "skills",
                "files",
            ],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([RefinePrompts.PREFIX, RefinePrompts.FORMAT])
        )
        messages = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )
        return messages

    def run_llm(self, qa_list, recent_history, full_history, skills, files):
        output = self.llm_chain(
            {
                "qa_list": {qa_list},
                "recent_history": {recent_history},
                "full_history": {full_history},
                "skills": {skills},
                "files": {files},
            }
        )
        return output


# class CurriculumAgent:
#     def __init__(
#         self,
#         tools=None,
#         model="gpt-4",
#         tools_model="gpt-4",
#         temp=0.1,
#         max_iterations=40,
#         qa_model_name="gpt-3.5-turbo",
#         qa_temperature=0,
#         request_timout=120,
#         api_key=None,
#         verbose=True,
#         ckpt_dir="ckpt",
#         resume=False,
#         mode="auto",
#         confirm_on=True,
#     ):
#         # create llm
#         llm = _make_llm(model, temp, verbose)
#         human_prompt = PromptTemplate(
#             template=curriculum_prompt,
#             input_variables=["recent_history", "full_history", "skills", "files"],
#         )
#         suffix = curriculum_format
#         human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
#         ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
#         system_message_prompt = SystemMessagePromptTemplate.from_template(
#             "\n\n".join([curriculum_prefix, curriculum_format])
#         )
#         prompt = ChatPromptTemplate.from_messages(
#             [system_message_prompt, human_message_prompt, ai_message_prompt]
#         )
#         self.llm = LLMChain(
#             llm=llm,
#             prompt=prompt,
#             callback_manager=StreamingStdOutCallbackHandler,
#         )

#         # may add hybrid mode later
#         assert mode in ["auto", "manual"], f"mode {mode} not supported"
#         self.mode = mode

#         self.ckpt_dir = ckpt_dir
#         if not os.path.exists(f"{ckpt_dir}/curriculum/"):
#             os.mkdir(f"{ckpt_dir}/curriculum/")

#         if resume:
#             print(
#                 f"\033[35mLoading Curriculum Agent from {ckpt_dir}/curriculum\033[0m")
#             )
#             with open(f"{self.ckpt_dir}/curriculum/completed_tasks.json", "w") as f1:
#                 self.completed_tasks = json.load(f1)

#             with open(f"{self.ckpt_dir}/curriculum/failed_tasks.json", "w") as f2:
#                 self.failed_tasks = json.load(f2)
#         else:
#             self.completed_tasks = []
#             self.failed_tasks = []

#     def render_observation(self, recent_history, files):
#         """
#         The observation of current event:
#         As of now, it only renders files, completed, and failed tasks.

#         """
#         files = ", ".join(files) if files else "None"
#         completed_tasks = (
#             ", ".join(self.completed_tasks) if self.completed_tasks else "None"
#         )
#         failed_tasks = ", ".join(self.failed_tasks) if self.failed_tasks else "None"

#         observation = {
#             # add other "recent_history" items
#             "files": f"Current list of files: {files}\n\n",
#             "completed_tasks": f"Completed tasks so far: {completed_tasks}\n\n",
#             "failed_tasks": f"Failed tasks that are too hard: {failed_tasks}\n\n",
#         }
#         return observation

# def propose_next_task(
#     self, recent_history, full_history, skills, files, max_retries=5
# ):
#         """
#         Given event, return task & context for the next step.

#         AUTO mode: the Curriculum agent will suggest next task.
#         If it's the first task, it will obtain PDB file for protein
#         specified by user

#         MANUAL mode: The user decides the next task

#         """

#         if len(self.completed_tasks) == 0 and self.mode == "auto":
#             print("Enter protein: ", end="", flush=True)
# timeout=10
# inputs, _, _ = select.select( [sys.stdin], [], [], timeout )
# if not inputs:
#     print(
#         f""" {timeout}-second timeout has passed. Proceeding
#         with fibronectin.
#         """
#     )
#     proteinname = "fibronectin"
# else:
#     proteinname = sys.stdin.readline().strip()

#             task = f"""Obtain PDB file for {proteinname} and map it to the
#             same name to be accessed later. Clean the PDB file as needed."""
#             return task

#         retries = 0
#         while retries < max_retries:
#             task = ""

#             if self.mode == "manual":
#                 task = input("Please enter task: ")

#             elif self.mode == "auto":
#                 response = self.llm.run(
#                     {
#                          "recent_history": recent_history,
#                         "full_history": full_history,
#                         "skills": skills,
#                         "files": files,
#                     }
#                 ).content

#                 # parse ai message
#                 try:
#                     task = ""
#                     for line in response.split("\n"):
#                         if line.startswith("Task:"):
#                             task = line[5:].replace(".", "").strip()
#                     assert task, "Task not found in Curriculum Agent response"
#                 except Exception as e:
#                     print(
#                         f"""\033[35mError parsing curriculum response: {e}.
#                         Trying again!\033[0m"""
#                     )
#             else:
#                 raise ValueError(f"Invalid curriculum agent mode: {self.mode}")

#             if task: # <- doesn't need this; we have assert check earlier
#                 if self.confirm_on:
#                     print(f"Task: {task}")
#                     #if input("Confirm? (y/n)").lower() not in ["y", ""]:
#                         # retries += 1
#                         # continue

#                     print("Confirm? (y/n): ",end="", flush=True)
#                     timeout=10 #seconds
#                     inputs, _, _ = select.select( [sys.stdin], [], [], timeout )
#                     if not inputs:
#                         print(
#                             f"""{timeout}-second timeout has passed. Proceeding
#                             with the task above."""
#                         )
#                     elif sys.stdin.readline().strip().lower() not in ["y", ""]:
#                         retries += 1
#                         continue

#                 return task

#             retries += 1

#         raise RuntimeError("Max retries reached, failed to propose a task.")

#     def update_progress(self, recent_history):
#         task = recent_history["task"]
#         if recent_history["success"]:
#             print(f"\033[35mCompleted task {task}.\033[0m")
#             self.completed_tasks.append(task)
#         else:
#             print(
#                 f"Failed to complete task {task}. Skipping to next task."
#             )
#             self.failed_tasks.append(task)

#         # clean up tasks and dump json files
#         self.clean_up_tasks()

#     def clean_up_tasks(self):
#         """
#         to remove any duplicates in completed tasks and
#         remove failed tasks that are already completed

#         dump json files of completed tasks, failed tasks, and
#         summary of failed tasks
#         """
#         updated_completed_tasks = []
#         updated_failed_tasks = self.failed_tasks

#         # dedupilcate but keep order
#         for task in self.completed_tasks:
#             if task not in updated_completed_tasks:
#                 updated_completed_tasks.append(task)

#         # remove completed tasks from failed tasks
#         for task in updated_completed_tasks:
#             while task in updated_failed_tasks:
#                 updated_failed_tasks.remove(task)

#         # get a summary of failed tasks & frequency
#         fail_summary = Counter(updated_failed_tasks)

#         self.completed_tasks = updated_completed_tasks
#         self.failed_tasks = updated_failed_tasks

#         with open(f"{self.ckpt_dir}/curriculum/completed_tasks.json", "w") as f1:
#             json.dump(self.completed_tasks, f1)

#         with open(f"{self.ckpt_dir}/curriculum/failed_tasks.json", "w") as f2:
#             json.dump(self.failed_tasks, f2)

#         with open(f"{self.ckpt_dir}/curriculum/failed_tasks_summary.json", "w") as f3:
#             json.dump(fail_summary, f3)
