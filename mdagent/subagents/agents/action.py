import io
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from mdagent.utils import PathRegistry

from .prompts import action_template

load_dotenv()


class Action:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-4",
        temp=0.1,
    ):
        llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_chain = LLMChain(llm=llm, prompt=action_template)
        self.path_registry = path_registry

    def _run(self, history, task, skills, args):
        # get files
        files = self.path_registry.list_path_names()
        # get skills
        return self.llm_chain(
            {
                "files": files,
                "task": task,
                "history": history,
                "skills": skills,
                "args": args,
                "init_dir": self.path_registry.ckpt_dir,
            }
        )["text"]

    # function that runs the code
    def _exec_code(self, python_code):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = captured_stdout = sys.stderr = io.StringIO()
        exec_context = {**globals(), **locals()}
        success = True
        try:
            exec(python_code, exec_context, exec_context)
            output = captured_stdout.getvalue()
        except Exception as e:
            success = False
            output = str(e)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        return success, output

    def _extract_code(self, output):
        # Regular expression to match a code block with optional 'python' keyword
        code_match = re.search(r"Code:\n```(?:python)?\n(.+?)\n```", output, re.DOTALL)

        if code_match:
            code = code_match.group(1)
            # Regular expression to extract the function name from the 'def' line
            fxn_match = re.search(r"def (\w+)\(", code)
            fxn_name = fxn_match.group(1) if fxn_match else None
            return code, fxn_name
        else:
            return None, None

    def _run_code(self, history, task, skills, args):
        # run agent
        output = self._run(history, task, skills, args)
        # extract code part
        code, fxn_name = self._extract_code(output)
        # run code
        success, code_output = self._exec_code(code)
        return success, code, fxn_name, code_output
