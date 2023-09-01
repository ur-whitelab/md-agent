import os, re
import sys, select
import json
import pickle
from typing import Optional
from langchain.tools import BaseTool

from ..base_tools import PathRegistry
from mdagent.subagents import SubAgents


class NewToolIterator:
    def __init__(
        self,
        agents,
        path_registry: Optional[PathRegistry],
        model="gpt-4",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        self.path_registry = path_registry
        self.action_agent = agents.action_agent
        #self.action_first_agent = agents.action_first_agent
        self.code_critic = agents.code_critic
        self.task_critic = agents.task_critic
        self.curriculum_agent = agents.refine_curriculum_agent
        self.skill_agent = agents.skill_agent

    def _add_to_history(
        self,
        existing_history,
        iter,
        task,
        context,
        code_history,
        output_history,
        critique_history,
        task_critique_history,
    ):
        # Initialize the output dictionary
        files_history = self.path_registry.list_path_names(True)
        if existing_history is None:
            existing_history = []

        # Initialize the output dictionary
        output_dict = {
            "iteration": iter,
            "task": task,
            "context": context,
            "code": code_history,
            "output": output_history,
            "files": files_history,
            "code critique": critique_history,
            "task_critique": task_critique_history,
        }
        # Append to the existing history
        output_json_string = json.dumps(output_dict, indent=4)
        existing_history.append(output_json_string)
        return existing_history

    def _save_failures(self, history, msg):
        if msg is None:
            # save to file
            with open("failed_history.json", "a") as f:
                f.write("\n", history, "\n")
            return "failed history saved to file"
        else:
            # save to file
            with open("failed_history.json", "a") as f:
                f.write("\n", msg, "\n")
            return None

    def _run_loop(self, task, context, recent_history, full_history):
        """
        this function just runs the iteration 1 time
        """
        code_success, code, code_output = self.action_agent._run_code(
            recent_history, full_history, task, context
        )
        if code_success is True:
            print("code succeeded, running task critic")
            # run task critic
            task_success, task_critique = self.task_critic_agent._run_task_critic(
                code, code_output, task, context, None
            )
        else:
            task_critique = None
            task_success = False

            # check if task is complete
            if task_success is True:
                print("task complete")
                return task_success, code, code_output, context, task, task_critique
            # otherwise, run code critic
        critique = self.code_critic_agent._run(code, code_output, task, context)
        return task_success, code, code_output, context, task, critique, task_critique

    def _run_iteration(
        self, run, task, context, iterations=5, failed=None, explanation=None
    ):
        # task is from curriculum
        # context is from curriculum
        self._save_failures(None, f"Run {run}")
        iter = 0
        success = False
        full_history = None
        recent_history = None
        while iter < iterations and success is False:
            if failed is not None:
                success, code, code_output = self.action_agent._run_code(
                    None, None, task, context, failed, explanation, "resume"
                )
                full_history = self._add_to_history(
                    None, iter, task, context, code, code_output, explanation, None
                )
            else:
                if iter > 0:
                    full_history = self._add_to_history(
                        full_history,
                        iter,
                        task,
                        context,
                        code,
                        code_output,
                        # todo: add these properly
                        # critique,
                        # task_critique,
                    )
                    recent_history = full_history[-1]
                (
                    success,
                    code,
                    output,
                    context,
                    task,
                    critique,
                    task_critique,
                ) = self._run_loop(task, context, recent_history, full_history)
            iter += 1
            failed = None
            # save to history
            if success:
                # update variables and save to file
                self._save_failures(full_history, None)
                # give successful code to tool manager
                return success, None
        # if max iterations reached without success, save failures to file
        print("max iterations reached, saving failed history to file")
        full_failed = self._add_to_history(
            full_history, iter, task, context, code, output, critique, task_critique
        )
        self._save_failures(full_failed, None)
        return success, full_failed

    def propose_refined_task(
        self,original_prompt, recent_history, full_history, skills, files, max_retries=5
    ):
        # ask curriculum agent to refine task in case the coder kept failing to produce 
        # a working code
        # manual mode is also available to manually enter task

        return "<NEW_TASK>"

    def add_new_tool(self, code):
        # skill agent store a new code in skill library & write new Langchain tool
        return "<TOOL_NAME>"


class ToolCreation(BaseTool):
    name = "ToolBuilder"
    description = """

    PUT DESCRIPTION HERE

    """
    subagents: Optional[SubAgents]
    path_registry: Optional[PathRegistry]

    def __init__(
        self, 
        subagents: Optional[SubAgents], 
        path_registry: Optional[PathRegistry]
    ) -> str:
        super().__init__()
        self.path_registry = path_registry
        self.agents = subagents

    def _run(self, query: str) -> str:
        """use the tool"""
        try:
            if self.path_registry is None:  # this should not happen
                return "Path registry not initialized"
            if self.agents is None:
                return "Agents for this tool are not initialized"
            newtool_result = NewToolIterator.run(self.agents, query)
            if newtool_result == None:
                return "Failed to write new code or tool"
            tool_name = newtool_result['tool_name']
            return f"""The new code is now stored as a new tool for next 
                MD-Agent prompt: {tool_name}"""
        except Exception as e:
            return f"Something went wrong. {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError("This tool does not support async")