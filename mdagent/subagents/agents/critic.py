from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from .prompts import critic_template

load_dotenv()


class Critic:
    def __init__(self, model="gpt-4-1106-preview", temp=0.1):
        llm = ChatOpenAI(
            temperature=temp,
            model=model,
            client=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.llm_chain = LLMChain(llm=llm, prompt=critic_template)

    def _run(self, code, task, code_output):
        output = self.llm_chain(
            {"code": code, "code_output": code_output, "task": task}
        )["text"]
        return output
