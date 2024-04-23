from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from mdagent.utils import PathRegistry

from .prompts import critic_template

load_dotenv()


class Critic:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        model="gpt-4-1106-preview",
        temp=0.1,
    ):
        llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.path_registry = path_registry
        self.llm_chain = LLMChain(llm=llm, prompt=critic_template)

    def _run(self, code, task, code_output):
        output = self.llm_chain(
            {
                "code": code,
                "code_output": code_output,
                "task": task,
                "init_dir": self.path_registry.ckpt_dir,
            }
        )["text"]
        return output
