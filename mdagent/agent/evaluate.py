import json
import os
import time

import pandas as pd

from .agent import MDAgent


# TODO: turn off verbose for MD-Agent -- verbose option doesn't work
# TODO: later, add write_to_notebooks option
class Evaluator:
    def __init__(self, eval_dir="evaluation_results"):
        # find root directory
        eval_path = eval_dir
        current_dir = os.getcwd()
        while current_dir != "/":
            if "setup.py" in os.listdir(current_dir):
                root_dir = os.path.abspath(current_dir)
                eval_path = os.path.join(root_dir, eval_dir)
                break
            else:
                current_dir = os.path.dirname(current_dir)
        self.base_dir = eval_path
        os.makedirs(self.base_dir, exist_ok=True)
        self.evaluations = []

    def create_agent(self, agent_params={}):
        """
        initializes MDAgent with given parameters

        Parameters:
        - agent_params (dict): dictionary of parameters to initialize MDAgent

        Returns:
        - initialized MDAgent object
        """
        # initialize MDAgent with given parameters.
        if agent_params is None:  # this shouldn't happen though
            agent_params = {}
        return MDAgent(**agent_params)

    def reset(self):
        """
        empties the evaluations list

        Parameters:
        - None

        Returns:
        - None
        """
        self.evaluations = []

    def save(self, filename="mega_eval", add_timestamp=True):
        """
        save all evaluations to a json file

        Parameters:
        - filename (str): name of the file to save evaluations to
        - add_timestamp (bool): whether to add a timestamp to the filename

        Returns:
        - None
        """
        if filename.endswith(".json"):
            filename = filename[:-5]
        if add_timestamp:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{filename}_{timestamp}"
        full_path = os.path.join(self.base_dir, f"{filename}.json")
        with open(full_path, "w") as f:
            json.dump(self.evaluations, f, indent=4)
        print(f"All evaluations saved to {full_path}.")

    def load(self, filename):
        """
        load past evaluations from a json file. Appends to the current evaluations list.

        Parameters:
        - filename (str): name of the file to load evaluations from

        Returns:
        - None
        """
        if not os.path.exists(filename):
            print(f"File {filename} not found. Please provide a valid file path.")
            return
        with open(filename, "r") as f:
            data = json.load(f)
        self.evaluations.extend(data)

    def _flatten_dict(self, d, sep="_"):
        """
        flattens evaluations dictionary up to 3 levels deep.
        Used in create_table method.

        Parameters:
        - d (dict): dictionary to flatten
        - sep (str): separator to use when flattening

        Returns:
        - flattened dictionary
        """
        flat_dict = {}
        for k1, v1 in d.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            flat_key = f"{k1}{sep}{k2}{sep}{k3}"
                            flat_dict[flat_key] = v3
                    else:
                        flat_key = f"{k1}{sep}{k2}"
                        flat_dict[flat_key] = v2
            else:
                flat_key = k1
                flat_dict[flat_key] = v1
        return flat_dict

    def _evaluate_all_steps(self, agent, user_prompt):
        """
        core function that evaluates while iterating every step of
        MDAgent's response to a user prompt. Evaluation details are
        saved to json file.
        Used in run_and_evaluate method.

        NOTE: This is not meant to be used directly. Use run_and_evaluate
        instead, since that method can capture exceptions and save to evaluations,
        which can be used to create a table.

        Parameters:
        - agent (MDAgent): MDAgent object
        - user_prompt (str): user prompt to evaluate

        Returns:
        - evaluation report (dict) containing details of the evaluation
        """
        num_steps = 0
        tools_used = {}
        tools_details = {}
        failed_steps = 0
        status_complete = "Unclear"
        step_start_time = start_time = time.time()
        for step in agent.iter(user_prompt):
            step_output = step.get("intermediate_step")
            if step_output:
                num_steps += 1
                action, observation = step_output[0]
                current_time = time.time()
                step_elapsed_time = current_time - step_start_time
                step_start_time = current_time
                tools_used[action.tool] = tools_used.get(action.tool, 0) + 1

                # determine success or failure from the first sentence of the output
                first_sentence = observation.split(".")[0]
                if "Failed" in first_sentence or "Error" in first_sentence:
                    status_complete = False
                    failed_steps += 1
                elif "Succeeded" in first_sentence:
                    status_complete = True
                else:
                    status_complete = "Unclear"

                tools_details[f"Step {num_steps}"] = {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": observation,
                    "status_complete": status_complete,
                    "step_elapsed_time (sec)": f"{step_elapsed_time:.3f}",
                    "timestamp_from_start (sec)": f"{current_time - start_time:.3f}",
                }
        final_output = step.get("output", "")
        if "Succeeded" in final_output.split(".")[0]:
            prompt_passed = True
        elif "Failed" in final_output.split(".")[0]:
            prompt_passed = False
        else:
            # If the last step output doesn't explicitly state "Succeeded" or "Failed",
            # determine the success of the prompt based on the previous step' status.
            prompt_passed = status_complete

        run_id = agent.run_id
        total_seconds = time.time() - start_time
        total_mins = total_seconds / 60
        agent_settings = {
            "llm": agent.llm.model_name,
            "agent_type": agent.agent_type,
            "tools_llm": agent.tools_llm.model_name,
            "use_memory": agent.use_memory,
        }
        print("\n----- Evaluation Summary -----")
        print("Run ID: ", run_id)
        print("Prompt success: ", prompt_passed)
        print(f"Total Steps: {num_steps+1}")
        print(f"Total Time: {total_seconds:.2f} seconds ({total_mins:.2f} minutes)")

        eval_report = {
            "agent_settings": agent_settings,
            "user_prompt": user_prompt,
            "prompt_success": prompt_passed,
            "total_steps": num_steps,
            "failed_steps": failed_steps,
            "total_time_seconds": f"{total_seconds:.3f}",
            "total_time_minutes": f"{total_mins:.3f}",
            "final_answer": final_output,
            "tools_used": tools_used,
            "tools_details": tools_details,
            "run_id": run_id,
        }
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"{agent.ckpt_dir}/evals", exist_ok=True)
        filename = f"{agent.ckpt_dir}/evals/individual_eval_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(eval_report, f, indent=4)
        return eval_report

    def run_and_evaluate(self, prompts, agent_params={}):
        """
        run and evaluate the agent with given parameters across multiple
        prompts. Appends to the evaluations list.

        Parameters:
        - prompts (list): list of prompts to evaluate
        - agent_params (dict): dictionary of parameters to initialize MDAgent

        Returns:
        - None
        """
        agent = self.create_agent(agent_params)
        for prompt in prompts:
            print(f"Evaluating prompt: {prompt}")
            try:
                eval_report = self._evaluate_all_steps(agent, prompt)
                eval_report["execution_success"] = True
                self.evaluations.append(eval_report)
            except Exception as e:
                agent_settings = {
                    "llm": agent.llm.model_name,
                    "agent_type": agent.agent_type,
                    "memory": agent.use_memory,
                }
                self.evaluations.append(
                    {
                        "agent_settings": agent_settings,
                        "prompt": prompt,
                        "execution_success": False,
                        "error_msg": f"{type(e).__name__}: {e}",
                    }
                )
                print(f"Error occurred while running MDAgent. {type(e).__name__}: {e}")

    def create_table(self, simple=True):
        """
        creates DataFrame table from evaluations list. Note that evaluations
        have to be loaded or generated first.

        Parameters:
        - simple (bool): whether to return a simplified table with fewer columns

        Returns:
        - DataFrame table of evaluations
        """
        evals = [self._flatten_dict(eval) for eval in self.evaluations]
        if not simple:
            return pd.DataFrame(evals)
        data = []
        for eval in evals:
            data.append(
                {
                    "LLM": eval.get("agent_settings_llm"),
                    "Agent Type": eval.get("agent_settings_agent_type"),
                    "User Prompt": eval.get("prompt"),
                    "Prompt Success": eval.get("prompt_success"),
                    "Execution Success": eval.get("execution_success"),
                    "Error Message": eval.get("error_msg"),
                    "Total Steps": eval.get("total_steps"),
                    "Failed Steps": eval.get("failed_steps"),
                    "Time (s)": eval.get("total_time_seconds"),
                    "Time (min)": eval.get("total_time_minutes"),
                    "Run ID": eval.get("run_id"),
                }
            )
        return pd.DataFrame(data)

    def automate(self, prompts, agent_params={}):
        """
        this automates the entire evaluation process for a given agent
        and prompts. It runs and evaluates, save the evaluations to a
        json file, and creates a table.

        Parameters:
        - prompts (list): list of prompts to evaluate
        - agent_params (dict): dictionary of parameters to initialize MDAgent

        Returns:
        - DataFrame table of evaluations
        """
        self.run_and_evaluate(prompts, agent_params)
        self.save()
        dataframe = self.create_table()
        return dataframe

    def automate_all(self, prompts, agent_params_list=None):
        """
        it automates the entire evaluation process for a list of agents.
        After evaluating all prompts with each agent, it saves the evaluations
        to a json file and creates a table containing all evaluations.

        Parameters:
        - prompts (list): list of prompts to evaluate
        - agent_params_list (list): list of dictionaries containing parameters
            to initialize MDAgent. If None, it will evaluate with default agents.

        Returns:
        - DataFrame table of evaluations
        """
        if agent_params_list is None:
            agent_params_list = [
                {
                    "agent_type": "OpenAIFunctionsAgent",
                    "model": "gpt-4-1106-preview",
                    "ckpt_dir": "ckpt_openaifxn_gpt4",
                },
                {
                    "agent_type": "Structured",
                    "model": "gpt-4-1106-preview",
                    "ckpt_dir": "ckpt_structured_gpt4",
                },
                {
                    "agent_type": "OpenAIFunctionsAgent",
                    "model": "gpt-3.5-turbo",
                    "ckpt_dir": "ckpt_openaifxn_gpt3.5",
                },
                {
                    "agent_type": "Structured",
                    "model": "gpt-3.5-turbo",
                    "ckpt_dir": "ckpt_structured_gpt3.5",
                },
            ]

        for agent in agent_params_list:
            self.run_and_evaluate(prompts, agent)
        self.save()
        dataframe = self.create_table()
        return dataframe
