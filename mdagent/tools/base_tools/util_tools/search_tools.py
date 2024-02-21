import os
import re
from langchain.base_language import BaseLanguageModel
import langchain
import paperqa
import paperscraper
from pypdf.errors import PdfReadError


def paper_scraper(search:str, pdir:str="query") -> dict:
    try:
        return paperscraper.search_papers(search, pdir=pdir)
    except KeyError:
        return {}
    
def paper_search(llm, query):
    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question"],
        template="""
        I would like to find scholarly papers to answer
        this question: {question}.
        'A search query that would bring up papers that can answer
        this question would be: '""",)
    
    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)
    if not os.path.isdir("./query"): #todo: move to ckpt
        os.mkdir("query/")

    search = query_chain.run(query)
    print("\nSearch:", search)
    papers = paper_scraper(search, pdir=f"query/{re.sub(' ', '', search)}")
    return papers


def scholar2result_llm(llm, query):
    """Useful to answer questions that require
    technical knowledge. Ask a specific question."""
    papers = paper_search(llm, query)
    if len(papers) == 0:
        return "Not enough papers found"
    docs = paperqa.Docs(llm=llm)
    not_loaded = 0
    for path, data in papers.items():
        try:
            docs.add(path, data["citation"])
        except (ValueError, FileNotFoundError, PdfReadError):
            not_loaded += 1

    print(f"\nFound {len(papers.items())} papers but couldn't load {not_loaded}")
    return docs.query(query).formatted_answer


class Scholar2ResultLLM:
    name = "Literature Search"
    description = (
        "Useful to answer questions that require technical ",
        "knowledge. Ask a specific question.",
    )
    llm: BaseLanguageModel

    def __init__(self, llm):
        self.llm = llm

    def _run(self, query) -> str:
        return scholar2result_llm(self.llm, query)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")