from prompts import action_critic_format, action_critic_prefix, action_critic_prompt
from action_agent import _make_llm
import json
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


class ActionCritic:
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
            template = action_critic_prompt,
            input_variables = ["task", "skills", "output", "errors"],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        ai_message_prompt = AIMessagePromptTemplate.from_template(suffix)
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            '\n\n'.join(
                [
                    action_critic_prefix,
                    action_critic_format
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
        self.llm_ = llm_chain
        return None

    def _run(self, skills, task, output, errors=None):
        self._create_llm()
        #get skills
        skills = None
        output = self.llm.run({"task": task, "skills": skills, "output": output, "errors": errors})
        return output
    
    def parse_action_critic(self,output):
        output = json.loads(output)
        return output
    
    def run_action_critic(self, task, code_output, retries=5):
        msg = None
        #get skills 
        skills = None
        for _ in range(retries):
            try: 
                critic_output = self._run(task, skills, code_output, errors=msg)
                parsed_output = self.parse_action_critic(critic_output)
                return code_output, parsed_output
            except json.JSONDecodeError:
                msg = "Error: Invalid JSON format. Please reformat your response."
        return code_output, None

