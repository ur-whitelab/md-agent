import logging
import os
import re
from typing import Optional

import langchain
import nest_asyncio
import paperqa
import paperscraper
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from pypdf.errors import PdfReadError

from mdagent.utils import PathRegistry


def configure_logging(path):
    # to log all runtime errors from paperscraper, which can be noisy
    log_file = os.path.join(path, "scraping_errors.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )


def paper_scraper(search: str, pdir: str = "query") -> dict:
    try:
        return paperscraper.search_papers(search, pdir=pdir)
    except KeyError:
        return {}


def paper_search(llm, query, path_registry, rounds=1):
    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question", "rounds"],
        template="""
        I would like to find papers to answer
        this question: {question}.
        Your response must have exact {rounds} of search query(ies),
        separated by comma, and each query must be at most 5 words long,
        nothing else. Fewer keywords is better.
        If a long protein name is given, use different known name or break
        down a long name for each query. For example,
        For `procollagen C-endopeptidase enchancer 1', your response would
        be: type 1 procollagen, procollagen C-proteinase,
        procollagen C-terminal proteinase.

        A list of search queries that would bring up papers that can answer
        this question would be: """,
    )

    path = f"{path_registry.ckpt_files}/query"
    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)
    if not os.path.isdir(path):
        os.mkdir(path)
    configure_logging(path)
    search = query_chain.run({"question": query, "rounds": rounds})
    search_list = [item.strip() for item in search.split(",")]
    papers = {}
    for s in search_list:
        print("\nSearch:", s)
        print("scraping...")
        try:
            papers.update(paper_scraper(s, pdir=f"{path}/{re.sub(' ', '', search)}"))
        except RuntimeError:
            print("Service limit of retries reached. Moved on.")
        print(len(papers), "papers scrapped so far.")
    return papers


def scholar2result_llm(llm, query, path_registry, k=5, max_sources=2, search_rounds=3):
    """Useful to answer questions that require
    technical knowledge. Ask a specific question."""
    papers = paper_search(llm, query, path_registry, rounds=search_rounds)
    if len(papers) == 0:
        print("No papers scraped. ")
        return (
            "Failed. Not enough papers found. "
            "Refine your query by using fewer or different keywords to broaden. "
            "Don't keep trying the exact same keywords over and over. "
        )
    print("Done scraping! Now processing the docs...")
    docs = paperqa.Docs(llm=llm.model_name)
    not_loaded = 0
    for path, data in papers.items():
        try:
            docs.add(path, data["citation"])
        except (ValueError, FileNotFoundError, PdfReadError):
            not_loaded += 1
        except Exception as e:
            # just to catch whatever else fail
            print(f"Cannot load this paper. {type(e).__name__}: {e}")
            not_loaded += 1

    print(
        f"\nScraped {len(papers)} papers"
        + (f" but couldn't load {not_loaded}" if not_loaded > 0 else "")
    )
    print("Now answering this question...")
    answer = docs.query(query, k=k, max_sources=max_sources).formatted_answer
    print(answer)
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
        nest_asyncio.apply()
        return scholar2result_llm(self.llm, query, self.path_registry)

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
