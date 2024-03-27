import json
from typing import Optional

import streamlit as st

from .subagent_setup import SubAgentInitializer, SubAgentSettings


class Iterator:
    def __init__(
        self,
        subagent_settings: Optional[SubAgentSettings],
        all_tools_string: Optional[str] = None,
        current_tools: Optional[dict] = None,
    ):
        if subagent_settings is None:
            raise ValueError("Subagent settings cannot be None")  # shouldn't happen
        self.path_registry = subagent_settings.path_registry
        self.memory = subagent_settings.memory
        self.ckpt_dir = subagent_settings.ckpt_dir
        self.all_tools_string = all_tools_string
        self.current_tools = current_tools

        # initialize agents
        initializer = SubAgentInitializer(subagent_settings)
        subagents = initializer.create_iteration_agents()
        self.action = subagents["action"]
        self.critic = subagents["critic"]
        self.skill = subagents["skill"]

    def _run_loop(self, task, full_history, skills):
        """
        this function just runs the iteration 1 time
        """
        critique = None
        print("\n\033[46m action agent is running, writing code\033[0m")
        st.markdown("action agent is running, writing code", unsafe_allow_html=True)
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
            full_history = self.memory._write_history_iterator(
                prompt=task,
                attempt_number=iter,
                code=code,
                output=code_output,
                critique=critique + suggestions,
                success=success,
            )
            if success:
                # give successful code to tool/skill manager
                print("\n\033[46mThe new code is complete, running skill agent\033[0m")
                st.markdown(
                    "The new code is complete, running skill agent",
                    unsafe_allow_html=True,
                )
                tool_name = self.skill.add_new_tool(fxn_name, code)
                return success, tool_name
            iter += 1

        # if max iterations reached without success, save failures to file
        print("\n\033[46m Max iterations reached, saving failed history to file\033[0m")
        st.markdown(
            "Max iterations reached, saving failed history to file",
            unsafe_allow_html=True,
        )
        tool_name = None
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
        full_history_string = self.memory.retrieve_recent_memory_iterator()

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
            "full_history": full_history_string,
            "skills": skills_string,
            "files": files_string,
            "current_tools": current_tools_string,
            "all_tools": self.all_tools_string,
        }
        return info
