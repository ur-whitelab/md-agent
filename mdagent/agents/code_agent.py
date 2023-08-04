from prompts import code_format, code_prefix, code_prompt, code_prefix_1, code_prompt_1
from . import PathRegistry
from action_agent import _make_llm
import json, sys, re
from typing import Optional
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

load_dotenv()

class CodeAgent:
    def __init__(
    self,
    path_registry: Optional[PathRegistry],
    model="gpt-4",
    temp=0.1,
    max_iterations=120,
    api_key=None,
    verbose=True,
    ):
        self.llm = _make_llm(model, temp, max_iterations)
        self.path_registry = path_registry
        
    def _create_prompt(self, resume):
        suffix = ""
        if resume: #if resume
            human_prompt = PromptTemplate(
                template = code_prompt,
                input_variables = ["recent_history", "full_history", "skills"],
            )
            prefix = code_prefix
        else: #if first iteration
            human_prompt = PromptTemplate(
                template = code_prompt_1,
                input_variables = ["files", "task", "context", "skills"],
            )
            prefix = code_prefix_1
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            '\n\n'.join(
                [
                    prefix,
                    code_format
                ]
            )
        )

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt, ai_message_prompt]
            )
        
    def _create_llm(self, resume):
        prompt = self._create_prompt(resume)
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            callback_manager=StreamingStdOutCallbackHandler,
            )
        self.llm_ = llm_chain
        return None

    def _run(self, recent_history, full_history, task, context):
        #get files
        files = self.path_registry.list_path_names(True)
        #get skills
        skills = None
        if full_history is None: #if first iteration
            return self.llm.run({"files": files, "task": task, "context": context, "skills": skills})
        return self.llm.run({"recent_history": recent_history, "full_history": full_history, "skills": skills})

    #function that runs the code
    def _exec_code(self, code):
        #incoming code should be a json string
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
            # group(1) corresponds to the first group of parentheses in the regex.
            # In this case, it is the section that matches any character (.) any number of times (+),
            # as long as it is between the "Code:\n```\n" and "\n```\n".
            code = match.group(1)
            return code
        else:
            return None
        
    def _run_code(self, recent_history, full_history, task, context):
        #create llm
        resume = full_history is not None
        self._create_llm(resume)
        #run agent
        output = self._run(recent_history, full_history, task, context)
        #extract code part
        code = self._extract_code(output)
        #run code
        success, code_output = self._exec_code(code)
        return success, code, code_output
        
