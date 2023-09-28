import json
import os
from typing import Optional

from mdagent.utils import PathRegistry

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
        code_critique_history,
        task_critique_history,
    ):
        # Initialize the output dictionary
        files_history = self.path_registry.list_path_names()
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
            "code critique": code_critique_history,
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
                f.write("\n" + history_string + "\n")
            return "failed history saved to file"
        else:
            # save to file
            with open(f"{self.ckpt_dir}/history/failed_history.json", "a") as f:
                f.write("\n" + msg + "\n")
            return None

    def _run_loop(self, task, context, recent_history, full_history, skills):
        """
        this function just runs the iteration 1 time
        """
        critique = None
        print("\n\033[46m action agent is running, writing code\033[0m")
        code_success, code, code_output = self.action_agent._run_code(
            recent_history,
            full_history,
            task,
            context,
            skills,
        )
        print("Code Output: ", code_output)
        if code_success is True:
            print("\n\033[46mcode succeeded, running task critic\033[0m")
            # run task critic
            task_success, task_critique = self.task_critic_agent._run_task_critic(
                code, code_output, task, context, None
            )
        else:
            task_critique = None
            task_success = False

        # check if task is complete
        if task_success is True:
            print("\n\033[46mtask complete\033[0m")
            return (
                task_success,
                code,
                code_output,
                context,
                task,
                critique,
                task_critique,
            )

        # otherwise, run code critic
        print("\n\033[46mtask failed, running code critic\033[0m")
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
        skills = self._pull_information()["skills"]
        while iter < iterations and success is False:
            # if failed is not None:
            #     success, code, code_output = self.action_agent._run_code(
            #         None, None, task, context, skills, failed, explanation, "resume"
            #     )
            #     full_history = self._add_to_history(
            #         None, iter, task, context, code, code_output, explanation, None
            #     )
            # else:
            # if iter > 0:
            #     full_history = self._add_to_history(
            #         full_history,
            #         iter,
            #         task,
            #         context,
            #         code,
            #         code_output,
            #         # TODO: add these properly
            #         # critique,
            #         # task_critique,
            #     )
            #     recent_history = full_history[-1]
            (
                success,
                code,
                code_output,
                context,
                task,
                code_critique,
                task_critique,
            ) = self._run_loop(task, context, recent_history, full_history, skills)

            # save to history
            full_history = self._add_to_history(
                full_history,
                iter,
                task,
                context,
                code,
                code_output,
                code_critique,
                task_critique,
            )
            recent_history = full_history[-1]
            if success:
                # update variables and save to file
                self._save_failures(full_history, None)

                # give successful code to tool/skill manager
                print("\n\033[46mThe new code is complete, running skill agent\033[0m")
                tool_name = self.skill_agent.add_new_tool(code, max_retries=5)
                return success, tool_name
            iter += 1

        # if max iterations reached without success, save failures to file
        print("\n\033[46m Max iterations reached, saving failed history to file\033[0m")
        tool_name = None
        full_failed = self._add_to_history(
            full_history,
            iter,
            task,
            context,
            code,
            code_output,
            code_critique,
            task_critique,
        )
        self._save_failures(full_failed, None)
        return success, tool_name

    # run da whole thing
    def run(self, task, user_prompt, max_task_refinement=1):
        for i in range(max_task_refinement + 1):
            if i > 0:
                # if not first step, propose a new task
                info = self._pull_information()
                print("\n\033[46mtask failed, running curriculum agent\033[0m")
                task = self.curriculum_agent.run(task, user_prompt, info, max_retries=3)

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

        # TODO: do include base tools
        skills = self.skill_agent.get_skills()
        if skills:
            skills_string = json.dumps(skills)
        else:
            skills_string = ""

        files = self.path_registry.list_path_names()
        if files:
            files_string = json.dumps(files)
        else:
            files_string = ""

        # TODO: include a list of packages we currently have/support

        info = {
            "recent_history": recent_history_string,
            "full_history": full_history_string,
            "skills": skills_string,
            "files": files_string,
        }
        return info
