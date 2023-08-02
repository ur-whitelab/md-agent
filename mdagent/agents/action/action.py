from ..mdagent.prompts.action import action_format, action_prefix, action_prompt
from ..mdagent.agent import _make_llm
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

class ActionAgent:
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
            template = action_prompt,
            input_variables = ["recent_history", "full_history", "skills"],
        )
        suffix = action_format
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            '\n\n'.join(
                [
                    action_prefix,
                    action_format
                ]
            )
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
        return llm_chain

    def _run(self, recent_history, full_history, skills):
        return self.llm.run({"recent_history": recent_history, "full_history": full_history, "skills": skills})
