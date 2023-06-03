import os
import langchain
from langchain import agents, prompts, chains, llms
from langchain.tools.python.tool import PythonREPLTool
import functools

from .general_tools import web_search, partial, scholar2result_llm

class MyPythonREPLTool(PythonREPLTool):
    @property
    def is_single_input(self):
        return True
    
class MDTools:
    def __init__(
        self,
        serp=None,
        openai=None,
        surrogate_llm="text-davinci-003",
        surrogate_llm_temp=0.1,
 ):
        self.serp_key = os.getenv("SERP_API_KEY") or serp
        self.openai_key = os.getenv("OPENAI_API_KEY") or openai
        self.surrogate_llm = surrogate_llm
        self.surrogate_llm_temp = surrogate_llm_temp

        # Initialize tool lists
        self.search_tools = self._search_tools()
        self.standard_tools = self._standard_tools()

        self.all_tools = (
            + self.search_tools
            + self.standard_tools
        )

        # Initialize standard tools
    def _standard_tools(self):
        """
        Standard tools:
        Tools directly imported from langchain: math, python-repl, etc.
        """
        sub_llm = langchain.OpenAI(
            temperature=self.surrogate_llm_temp, model_name=self.surrogate_llm
        )

        self.standard_tools = agents.load_tools(
            [
                "python_repl",
                "human"
            ], sub_llm)

        return self.standard_tools
    
    def _search_tools(self):
        """
        Search tools:
        Tools for extracting knowledge from the internet, and knowledge distillation with llms.
        """

        # Define an llm for search
        llm_search = langchain.OpenAI(
            temperature=0.05,
            model_kwargs={"stop": ['"']},
        )

        scholar2result = functools.partial(scholar2result_llm, llm_search)     

        search_tools = [
            agents.Tool(
                name="LiteratureSearch",
                func=scholar2result,
                description=(
                    "Input a specific question, returns an answer from literature search. "
                ),
            ),
            agents.Tool(
                name="WebSearch",
                func=web_search,
                description=(
                    "Input search query, returns snippets from web search. "
                    "Prefer LitSearch tool over this tool, except for simple questions."),
            ),
        ]
