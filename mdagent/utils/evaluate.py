import json
import os
import time

import pandas as pd

from mdagent import MDAgent


class Evaluator:
    def __init__(self, base_dir="evaluation_results"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.evaluations = []

    def create_agent(self, agent_params={}):
        # initialize MDAgent with given parameters.
        return MDAgent(**agent_params)

    def reset(self):
        self.evaluations = []

    def _evaluate_single_step(self, step_output):
        pass

    def _evaluate_all_steps(self, agent, user_prompt):
        num_steps = 0
        tools_used = {}
        tools_details = {}
        failed_steps = 0
        status = "Unclear"
        step_start_time = start_time = time.time()
        for step in agent.iter(user_prompt):
            step_output = step.get("intermediate_step")
            # self._evaluate_single_step(step_output)'
            if step_output:
                num_steps += 1
                action, observation = step_output[0]
                current_time = time.time()
                step_elapsed_time = current_time - step_start_time
                step_start_time = current_time
                tools_used[action.tool] = tools_used.get(action.tool, 0) + 1

                # Determine success or failure from the first sentence of the output
                first_sentence = observation.split(".")[
                    0
                ]  # Assuming sentences end with '.'
                if "Failed" in first_sentence or "Error" in first_sentence:
                    status = "Failed"
                elif "Succeeded" in first_sentence:
                    status = "Succeeded"
                else:
                    status = "Unclear"

                tools_details[f"Step {num_steps}"] = {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": observation,
                    "status": status,  # Include success/failure status
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
            prompt_passed = status

        run_id = str(getattr(step.get("__run"), "run_id", None))
        print("Run ID: ", run_id)
        total_seconds = time.time() - start_time
        total_mins = total_seconds / 60
        agent_settings = {
            "llm": agent.llm.model_name,
            "agent_type": agent.agent_type,
            "resume": agent.subagents_settings.resume,
            "learn": not agent.skip_subagents,
            "curriculum": agent.subagents_settings.curriculum,
            # "memory": agent.subagents_settings.memory,
        }
        print("\n----- Evaluation Summary -----")
        print(f"Total Steps: {num_steps+1}")
        print(f"Total Time: {total_seconds:.2f} seconds ({total_mins:.2f} minutes)")

        summary = {
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
        os.makedirs(f"{agent.ckpt_dir}/eval", exist_ok=True)
        filename = f"{agent.ckpt_dir}/eval/eval_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(summary, f, indent=4)

        brief_summary = {
            "agent_settings": agent_settings,
            "prompt_success": prompt_passed,
            "total_steps": num_steps,
            "failed_steps": failed_steps,
            "total_time_seconds": f"{total_seconds:.3f}",
            "run_id": run_id,
            "final_answer": final_output,
        }
        return brief_summary

    def run_evaluation(self, prompts, agent_params=None, same_ckpt=False):
        """
        Evaluate the agent with given parameters across multiple prompts.
        """
        agent = self.create_agent(agent_params)
        # TODO: get number of skills from skills.json if it exists in each ckpt

        for count, prompt in enumerate(prompts):
            print(f"Evaluating prompt: {prompt}")
            if not same_ckpt:
                agent.ckpt_dir = (
                    f"{agent.ckpt_dir}_{count+1}"  # unqiue ckpt dir for each prompt
                )
            try:
                brief_summary = self._evaluate_all_steps(agent, prompt)
                self.evaluations.append(
                    {
                        "execution_success": True,
                        "agent_settings": brief_summary["agent_settings"],
                        "prompt": prompt,
                        "prompt_success": brief_summary["prompt_success"],
                        "summary": brief_summary,
                    }
                )
            except Exception as e:
                agent_settings = {
                    "llm": agent.llm.model_name,
                    "agent_type": agent.agent_type,
                    "resume": agent.subagents_settings.resume,
                    "learn": not agent.skip_subagents,
                    "curriculum": agent.subagents_settings.curriculum,
                    "memory": agent.subagents_settings.memory,
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

    def summarize_and_save(self, filename="mega_eval"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.base_dir, f"{filename}_{timestamp}.json")
        summary = {
            "total_evaluations": len(self.evaluations),
            "evaluations": self.evaluations,
        }
        with open(filename, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"All evaluations saved to {filename}.")

    def create_prompt_table(self, table_filename="eval_table"):
        """
        Creates a summary table using pandas DataFrame
        and saves it as a JSON file with a timestamped filename.
        """
        data = [
            {
                "Prompt Number": i + 1,
                # "Prompt": eval["prompt"],
                "Execution Success": eval["execution_success"],
                "Total Steps": eval["summary"]["total_steps"],
                "Failed Steps": eval["summary"]["failed_steps"],
                "Prompt Success": eval["summary"]["prompt_success"],
                "Total Time (Seconds)": eval["summary"]["total_time_seconds"],
            }
            for i, eval in enumerate(self.evaluations)
            if eval["execution_success"]
        ]

        # TODO: deal with failed evaluations

        df = pd.DataFrame(data)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.base_dir, f"{table_filename}_{timestamp}.json")
        df.to_json(filename, orient="records", indent=4)
        print(f"Summary table saved to {filename}.")

        return df

    def table_to_latex(self, df, filename="eval_table"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.base_dir, f"{filename}_{timestamp}.tex")
        df.to_latex(filename, index=False)

    def automate(self, prompts, agent_params=None):
        self.run_evaluation(prompts, agent_params)
        self.summarize_and_save()
        dataframe = self.create_prompt_table()
        return dataframe

    def automate_all(self, prompts, agent_params_list=None):
        # automate evals for one prompt for different agent types
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
            self.run_evaluation(prompts, agent)
        self.summarize_and_save()
        # TODO: add a function to create a agent-focused table


# Example usage

# # Initialize the Evaluator
# evaluator = Evaluator()

# # Set the base checkpoint directory when you're ready
# evaluator.set_base_ckpt_dir("ckpt/evaluations")

# # Define different sets of parameters to evaluate
# agent_configs = [
#     {"agent_type": "OpenAIFunctionsAgent", "model": "gpt-4-1106-preview"},
#     {"agent_type": "Structured", "model": "gpt-4-1106-preview"},
#     # Add more configurations as needed
# ]

# # Define a list of prompts you want to evaluate
# prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

# # Run evaluations for each configuration
# for config in agent_configs:
#     evaluator.run_evaluation(agent_params=config, prompts=prompts)

# # Summarize and save the evaluations
# evaluator.summarize_and_save(summary_filename="evaluation_summary")
