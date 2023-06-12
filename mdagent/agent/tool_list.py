import os


class MyPythonREPLTool:
    @property
    def is_single_input(self):
        return True

    # Note that the import for PythonREPLTool is within the constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MDTools:
    def __init__(
        self,
        serp=None,
        openai=None,
        pqa=None,
        surrogate_llm="text-davinci-003",
        surrogate_llm_temp=0.1,
    ):
        self.serp_key = os.getenv("SERP_API_KEY") or serp
        self.openai_key = os.getenv("OPENAI_API_KEY") or openai
        self.pqa_key = os.getenv("PQA_API_KEY") or pqa
        self.surrogate_llm = surrogate_llm
        self.surrogate_llm_temp = surrogate_llm_temp

        # Initialize tool lists
        self.search_tools = self._search_tools()
        self.standard_tools = self._standard_tools()

        self.all_tools = self.search_tools + self.standard_tools

        # Initialize standard tools

    def _standard_tools(self):
        """
        Standard tools:
        Tools directly imported from langchain: math, python-repl, etc.
        """
        import langchain
        from langchain import agents

        sub_llm = langchain.OpenAI(
            temperature=self.surrogate_llm_temp, model_name=self.surrogate_llm
        )

        self.standard_tools = agents.load_tools(
            ["python_repl", "human", "llm-math"], sub_llm
        )

        return self.standard_tools

    def _search_tools(self):
        """
        Search tools:
        Tools for extracting knowledge from the internet
        and knowledge distillation with llms.
        """
        from langchain import agents

        from ..general_tools import Scholar2ResultLLM

        if self.pqa_key is not None:
            pqa_result = Scholar2ResultLLM(self.pqa_key)

        search_tools = [
            agents.Tool(
                name="LiteratureSearch",
                func=pqa_result.query,
                description=(
                    "Input a specific question,"
                    "returns an answer from literature search. "
                ),
            )
        ]
        return search_tools
