import json
import re
import sys
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

from . import (
    PathRegistry,
    code_format,
    code_prefix,
    code_prefix_1,
    code_prompt,
    code_prompt_1,
    make_llm,
)

load_dotenv()


class Action:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-4",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        self.llm = make_llm(model, temp, max_iterations)
        self.path_registry = path_registry

    def _create_prompt(self, version):
        suffix = ""
        if version == "resume":  # if resume
            human_prompt = PromptTemplate(
                template=code_prompt,
                input_variables=["recent_history", "full_history", "skills"],
            )
            prefix = code_prefix
        elif version == "first":  # if first iteration
            human_prompt = PromptTemplate(
                template=code_prompt_1,
                input_variables=["files", "task", "context", "skills"],
            )
            prefix = code_prefix_1
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([prefix, code_format])
        )

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
        )

    def _create_llm(self, version):
        prompt = self._create_prompt(version)
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            callback_manager=StreamingStdOutCallbackHandler,
        )
        self.llm_ = llm_chain
        return None

    def _run(
        self, version, recent_history, full_history, task, context, failed, explanation
    ):
        # get files
        files = self.path_registry.list_path_names(True)
        # get skills
        skills = None
        if version == "resume":  # if resume
            return self.llm.run(
                {
                    "recent_history": recent_history,
                    "full_history": full_history,
                    "skills": skills,
                }
            )
        elif version == "first":  # if first iter
            return self.llm.run(
                {"files": files, "task": task, "context": context, "skills": skills}
            )

    # function that runs the code
    def _exec_code(self, code):
        # incoming code should be a json string
        # Load the JSON string and extract the Python code
        data = json.loads(code)
        python_code = data["code"]

        # Redirect stdout and stderr to capture the output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = captured_stdout = sys.stderr = sys.StringIO()
        success = True
        try:
            exec(python_code)
            output = captured_stdout.getvalue()
        except Exception as e:
            success = False
            output = str(e)
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        return success, output

    def _extract_code(self, output):
        match = re.search(r"Code:\n```\n(.+?)\n```\n", output, re.DOTALL)
        if match:
            code = match.group(1)
            return code
        else:
            return None

    def _run_code(
        self,
        recent_history,
        full_history,
        task,
        context,
        failed=None,
        explanation=None,
        version="resume",
    ):
        if failed is None and recent_history is None:
            version = "first"
        # create llm
        self._create_llm(version)
        # run agent
        output = self._run(
            self,
            version,
            recent_history,
            full_history,
            task,
            context,
            failed,
            explanation,
        )
        # extract code part
        code = self._extract_code(output)
        # run code
        success, code_output = self._exec_code(code)
        return success, code, code_output
