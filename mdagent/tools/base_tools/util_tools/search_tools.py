import os
import re
from typing import Optional

import langchain
import paperqa
import paperscraper
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from pypdf.errors import PdfReadError

from mdagent.utils import PathRegistry


def paper_scraper(search: str, pdir: str = "query") -> dict:
    try:
        return paperscraper.search_papers(search, pdir=pdir)
    except KeyError:
        return {}


def paper_search(llm, query, path_registry):
    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question"],
        template="""
        I would like to find scholarly papers to answer
        this question: {question}. Your response must be at
        most 10 words long.
        'A search query that would bring up papers that can answer
        this question would be: '""",
    )

    path = f"{path_registry.ckpt_files}/query"
    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)
    if not os.path.isdir(path):
        os.mkdir(path)
    search = query_chain.run(query)
    print("\nSearch:", search)
    papers = paper_scraper(search, pdir=f"{path}/{re.sub(' ', '', search)}")
    return papers


def scholar2result_llm(llm, query, path_registry, k=5, max_sources=2):
    """Useful to answer questions that require
    technical knowledge. Ask a specific question."""
    papers = paper_search(llm, query, path_registry)
    if len(papers) == 0:
        return "Not enough papers found"
    docs = paperqa.Docs(llm=llm.model_name)
    not_loaded = 0
    for path, data in papers.items():
        try:
            docs.add(path, data["citation"])
        except (ValueError, FileNotFoundError, PdfReadError):
            not_loaded += 1

    print(
        f"\nFound {len(papers)} papers"
        + (f" but couldn't load {not_loaded}" if not_loaded > 0 else "")
    )
    answer = docs.query(query, k=k, max_sources=max_sources).formatted_answer
    return answer


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = (
        "Useful to answer questions that require technical "
        "knowledge. Ask a specific question."
    )
    llm: BaseLanguageModel = None
    path_registry: Optional[PathRegistry]

    def __init__(self, llm, path_registry):
        super().__init__()
        self.llm = llm
        self.path_registry = path_registry

    def _run(self, query) -> str:
        return scholar2result_llm(self.llm, query, self.path_registry)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
