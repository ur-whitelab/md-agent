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

from ...mainagent import _make_llm
from ..prompts import code_critic_format, code_critic_prefix, code_critic_prompt

load_dotenv()


class CodeCriticAgent:
    def __init__(
        self,
        model="gpt-4",
        temp=0.1,
        max_iterations=120,
        api_key=None,
        verbose=True,
    ):
        self.llm = _make_llm(model, temp, max_iterations)

    def _create_prompt(self):
        suffix = ""
        human_prompt = PromptTemplate(
            template=code_critic_prompt,
            input_variables=["code", "code_output", "task", "context"],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "\n\n".join([code_critic_prefix, code_critic_format])
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

    def _run(self, src, task, context, code_output):
        self._create_llm()
        output = self.llm.run(
            {"code": src, "code_output": code_output, "task": task, "context": context}
        )
        return output
