import json
import os
from typing import Optional

from ..tools import PathRegistry
from .subagent_setup import SubAgentInitializer, SubAgentSettings


class Iterator:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        subagent_settings: Optional[SubAgentSettings],
    ):
        self.path_registry = path_registry
        if subagent_settings is None:
            raise ValueError("Subagent settings cannot be None")  # shouldn't happen
        self.ckpt_dir = subagent_settings.ckpt_dir
        os.makedirs(f"{self.ckpt_dir}/history/", exist_ok=True)

        # initialize agents
        initializer = SubAgentInitializer(subagent_settings)
        subagents = initializer.create_iteration_agents()
        self.action_agent = subagents["action"]
        self.code_critic_agent = subagents["code_critic"]
        self.curriculum_agent = subagents["refining_curriculum"]
        self.skill_agent = subagents["skill"]
        self.task_critic_agent = subagents["task_critic"]

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
            with open(f"{self.ckpt_dir}/history/failed_history.json", "a") as f:
                history_string = json.dumps(history)
                f.write("\n", history_string, "\n")
            return "failed history saved to file"
        else:
            # save to file
            with open(f"{self.ckpt_dir}/history/failed_history.json", "a") as f:
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

    def _run_iterations(
        self, run, task, context, iterations=5, failed=None, explanation=None
    ):
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

                # give successful code to tool/skill manager
                tool_name = self.skill_agent.add_new_tool(code, max_retries=5)
                return success, tool_name

        # if max iterations reached without success, save failures to file
        print("max iterations reached, saving failed history to file")
        full_failed = self._add_to_history(
            full_history, iter, task, context, code, output, critique, task_critique
        )
        self._save_failures(full_failed, None)
        return success, tool_name

    # run da whole thing
    def run(self, task, user_prompt, max_task_refinement=1):
        for i in range(max_task_refinement + 1):
            if i > 0:
                # refine and propose a new task
                info = self._pull_information()
                task = self.curriculum_agent.run(task, user_prompt, info, max_retries=3)

            # run iterations to get the new code
            success, tool_name = self._run_iterations(
                i, task, user_prompt, iterations=5
            )
            if success:
                return tool_name
        return None

    def _pull_information(self):
        # pull info of strings to pass to llm agents
        recent_history_string = ""
        full_history_string = ""
        if os.path.exists(f"{self.ckpt_dir}/history/failed_history.json"):
            with open(f"{self.ckpt_dir}/history/failed_history.json", "r") as f:
                full_history_string = f.read()
                lines = full_history_string.splitlines()
                recent_history_string = lines[-1] if lines else None

        skills = self.skill_agent.get_skills()
        skills_string = json.dumps(skills)

        files = self.path_registry.list_path_names(True)
        files_string = json.dumps(files)

        info = {
            "recent_history": recent_history_string,
            "full_history": full_history_string,
            "skills": skills_string,
            "files": files_string,
        }
        return info
