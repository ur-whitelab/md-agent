import json
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

from mdagent.subagents import (
    task_critic_format, 
    task_critic_prefix, 
    task_critic_prompt)
from mdagent.mainagent import _make_llm

from mdagent.tools import PathRegistry

load_dotenv()


class TaskCritic:
    def __init__(
        self,
        path_registry=Optional[PathRegistry],
        model="gpt-4",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        self.llm = _make_llm(model, temp, max_iterations)
        self.path_regisry = path_registry

    def _create_prompt(self):
        suffix = ""
        human_prompt = PromptTemplate(
            template=task_critic_prompt,
            input_variables=[
                "files",
                "code",
                "code_output",
                "task",
                "context",
                "additional_information",
            ],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([task_critic_prefix, task_critic_format])
        )
        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )

    def _create_llm(self):
        prompt = self._create_prompt()
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            callback_manager=StreamingStdOutCallbackHandler,
        )
        self.llm_ = llm_chain
        return None

    def _run(self, code, code_output, task, context, additional_information):
        # get files
        files = self.path_registry.list_path_names(True)
        output = self.llm.run(
            {
                "files": files,
                "code": code,
                "code_output": code_output,
                "task": task,
                "context": context,
                "additional_information": additional_information,
            }
        )
        return output

    def _parse_critic_output(self, critic_output):
        parsed_output = json.loads(critic_output)
        # Extract the success boolean
        success = parsed_output.get("success", None)
        # Check if it's a boolean value first
        if isinstance(success, bool):
            return success, parsed_output.get("critique", None)

        # If not a boolean, make it boolean
        if isinstance(success, str):
            success_lower = success.lower()
            if "true" in success_lower:
                success = True
            elif "false" in success_lower:
                success = False
            else:
                raise ValueError(f"Invalid success value: {success}")
            return success, parsed_output.get("critique", None)

    def _run_task_critic(self, code, code_output, task, context, msg):
        # make llm
        self._create_llm()
        # running task critic, doing this in a loop to handle an error one time
        for _ in range(2):  # Two attempts
            try:
                full_out = self._run(code, code_output, task, context, msg)
                success, task_critique = self._parse_critic_output(full_out)
                # Break out of the loop if successful
                break
            except ValueError:
                # Raised if the success value is invalid
                msg = """Please ensure that your
                output matches the formatting requirements."""
                # try again with additional message
        else:  # Executed if both attempts failed
            success = False
            task_critique = None
        return success, task_critique
