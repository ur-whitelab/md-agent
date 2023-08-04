import langchain
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from rmrkl import ChatZeroShotAgent
from langchain import agents

from prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, SUFFIX

load_dotenv()

def _make_llm(model, temp, verbose):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=True if verbose else False,
            callbacks=[StreamingStdOutCallbackHandler()] if verbose else [None],
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm

class ActionAgent:
    def __init__(
        self,
        tools=None,
        model="gpt-4",
        tools_model="gpt-4",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
    ):
        self.llm = _make_llm(model, temp, verbose)
        
    def make_tools(self):
        tools_llm = _make_llm(self.tools_model, self.temp, self.verbose)
        all_tools = agents.load_tools(["python_repl", "human", "llm-math"], tools_llm)
        #add in tools from tool library
        return all_tools

        # Initialize agent
    def init_agent(self, tools):
        self.agent = ChatZeroShotAgent.from_llm_and_tools(
            self.llm,
            tools=tools,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            question_prompt=QUESTION_PROMPT,
        )
        return None

    def run(self, prompt):
        #get tools
        status = False
        tools = self.make_tools()
        self.init_agent(tools)
        try:
            outputs = self.agent({"input": prompt})
            # Parse long output (with intermediate steps)
            intermed = outputs["intermediate_steps"]

            final = ""
            for step in intermed:
                final += f"Thought: {step[0].log}\n" f"Observation: {step[1]}\n"
            final += f"Final Answer: {outputs['output']}"

            return status, final
        except Exception as e:
            return status, str(e)
