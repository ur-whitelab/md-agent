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
        all_tools_string: Optional[str] = None,
        current_tools: Optional[dict] = None,
    ):
        self.path_registry = path_registry
        if subagent_settings is None:
            raise ValueError("Subagent settings cannot be None")  # shouldn't happen
        self.ckpt_dir = subagent_settings.ckpt_dir
        self.all_tools_string = all_tools_string
        self.current_tools = current_tools
        os.makedirs(f"{self.ckpt_dir}/history/", exist_ok=True)

        # initialize agents
        initializer = SubAgentInitializer(subagent_settings)
        subagents = initializer.create_iteration_agents()
        self.action = subagents["action"]
        self.critic = subagents["critic"]
        self.skill = subagents["skill"]

    def _add_to_history(
        self,
        existing_history,
        iter,
        task,
        code_history,
        output_history,
        critique,
        suggestions,
    ):
        # Initialize the output dictionary
        files_history = self.path_registry.list_path_names()
        if existing_history is None:
            existing_history = []

        # Initialize the output dictionary
        output_dict = {
            "iteration": iter,
            "task": task,
            "code": code_history,
            "output": output_history,
            "files": files_history,
            "critique": critique,
            "suggestions": suggestions,
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

    def _run_loop(self, task, full_history, skills):
        """
        this function just runs the iteration 1 time
        """
        critique = None
        print("\n\033[46m action agent is running, writing code\033[0m")
        success, code, fxn_name, code_output = self.action._run_code(
            full_history, task, skills
        )
        print("\nCode Output: ", code_output)
        critique = self.critic._run(code, task, code_output)
        critique = critique.replace("```json", "").replace("```", "").strip()
        critique_full = json.loads(critique)
        task_relevance = critique_full["task_relevance"]
        critique = critique_full["critique"]
        suggestions = critique_full["suggestions"]
        if task_relevance and success:
            success = True
        else:
            success = False
        return success, code, fxn_name, code_output, task, critique, suggestions

    def _run_iterations(self, run, task):
        iterations = 5
        self._save_failures(None, f"Run {run}")
        iter = 0
        success = False
        full_history = None
        skills = self._pull_information()["skills"]
        while iter < iterations and success is False:
            (
                success,
                code,
                fxn_name,
                code_output,
                task,
                critique,
                suggestions,
            ) = self._run_loop(task, full_history, skills)

            # save to history
            full_history = self._add_to_history(
                full_history,
                iter,
                task,
                code,
                code_output,
                critique,
                suggestions,
            )
            if success:
                # update variables and save to file
                self._save_failures(full_history, None)

                # give successful code to tool/skill manager
                print("\n\033[46mThe new code is complete, running skill agent\033[0m")
                tool_name = self.skill.add_new_tool(fxn_name, code)
                return success, tool_name
            iter += 1

        # if max iterations reached without success, save failures to file
        print("\n\033[46m Max iterations reached, saving failed history to file\033[0m")
        tool_name = None
        full_failed = self._add_to_history(
            full_history,
            iter,
            task,
            code,
            code_output,
            critique,
            suggestions,
        )
        self._save_failures(full_failed, None)
        return success, tool_name

    # run da whole thing
    def run(self, task, user_prompt):
        # info = self._pull_information() # if you want to pass any of these info
        success, tool_name = self._run_iterations(
            task,
            user_prompt,
        )
        if success:
            return tool_name
        else:
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

        skills = self.skill.get_skills()
        if skills:
            skills_string = json.dumps(skills)
        else:
            skills_string = ""

        files = self.path_registry.list_path_names()
        if files:
            files_string = json.dumps(files)
        else:
            files_string = ""

        current_tools_string = ""
        if self.current_tools:
            current_tools_string = json.dumps(self.current_tools)

        info = {
            "recent_history": recent_history_string,
            "full_history": full_history_string,
            "skills": skills_string,
            "files": files_string,
            "current_tools": current_tools_string,
            "all_tools": self.all_tools_string,
        }
        return info
