from ..mdagent.prompts.action_first import action_format, action_prefix_1, action_prompt_1
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

class FirstActionAgent:
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
#if first iteration
        human_prompt = PromptTemplate(
            template = action_prompt_1,
            input_variables = ["files", "task", "context", "skills"],
        )
        suffix = action_format
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            '\n\n'.join(
                [
                    action_prefix_1,
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

    def _run(self, task, context, files, skills):
        return  self.llm.run({"files": files, "task": task, "context": context, "skills": skills})
        