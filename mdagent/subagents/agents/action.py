import io
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from mdagent.utils import PathRegistry

from .examples import few_shot_prompt
from .prompts import action_template_1, action_template_2, md_expert_template

load_dotenv()


class Action:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-4",
        temp=0.1,
    ):
        self.llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_chain_1 = LLMChain(llm=self.llm, prompt=action_template_1)
        self.llm_chain_2 = LLMChain(llm=self.llm, prompt=action_template_2)
        self.llm_chain_3 = None
        self.path_registry = path_registry

    def _run_action_writer_1(self, history, task, skills, args, code=""):
        # get files
        files = self.path_registry.list_path_names_and_descriptions()
        # get skills
        return self.llm_chain_1(
            {
                "files": files,
                "task": task,
                "history": history,
                "skills": skills,
                "code": code,
                "args": args,
                "init_dir": self.path_registry.ckpt_dir,
            }
        )["text"]

    def _run_action_writer_paths(self, task, code, args):
        # get files
        files = self.path_registry.list_path_names_and_descriptions()
        # get skills
        return self.llm_chain_2(
            {
                "files": files,
                "task": task,
                "code": code,
                "args": args,
            }
        )["text"]

    def _run_md_expert_writer(self, task, code, new_task, args):
        files = self.path_registry.list_path_names_and_descriptions()
        if not self.llm_chain_3 or new_task:
            _few_shot_examples = (
                few_shot_prompt.format(input=task)
                .replace("Human:", " Task Example: ")
                .replace("AI:", " Generation Example:")
            )
            md_expert_prompt = md_expert_template.format(
                few_shot_prompt=_few_shot_examples
            )
            self.llm_chain_3 = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["files", "task", "code"], template=md_expert_prompt
                ),
            )

        return self.llm_chain_3(
            {
                "files": files,
                "task": task,
                "code": code,
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

    def _run_code(self, history, task, skills, args, new_task, code=""):
        # run agent
        output = self._run_action_writer_1(history, task, skills, args, code)
        # extract code part
        code, _ = self._extract_code(output)
        # run code
        ###here we're adding the second llm refinement
        output = self._run_action_writer_paths(task, code, args)
        code, _ = self._extract_code(output)

        output = self._run_md_expert_writer(task, code, new_task, args)
        code, fxn_name = self._extract_code(output)
        ### here we change paths to use the path registry for saving and loading
        success, code_output = self._exec_code(code)
        return success, code, fxn_name, code_output
