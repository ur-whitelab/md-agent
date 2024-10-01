from typing import Optional

import nest_asyncio
import paperqa
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool

from mdagent.utils import PathRegistry

# from pypdf.errors import PdfReadError


# def configure_logging(path):
#     # to log all runtime errors from paperscraper, which can be VERY noisy
#     log_file = os.path.join(path, "scraping_errors.log")
#     logging.basicConfig(
#         filename=log_file,
#         level=logging.ERROR,
#         format="%(asctime)s:%(levelname)s:%(message)s",
#     )


# def paper_scraper(search: str, pdir: str = "query") -> dict:
#     try:
#         return paperscraper.search_papers(search, pdir=pdir)
#     except KeyError:
#         return {}


# def paper_search(llm, query, path_registry):
#     prompt = langchain.prompts.PromptTemplate(
#         input_variables=["question"],
#         template="""
#         I would like to find scholarly papers to answer
#         this question: {question}. Your response must be at
#         most 10 words long.
#         'A search query that would bring up papers that can answer
#         this question would be: '""",
#     )

#     path = f"{path_registry.ckpt_files}/query"
#     query_chain = prompt | llm | StrOutputParser()
#     if not os.path.isdir(path):
#         os.mkdir(path)
#     configure_logging(path)
#     search = query_chain.invoke(query)
#     print("\nSearch:", search)
#     papers = paper_scraper(search, pdir=f"{path}/{re.sub(' ', '', search)}")
#     return papers


# def scholar2result_llm(llm, query, path_registry, k=5, max_sources=2):
#     """Useful to answer questions that require
#     technical knowledge. Ask a specific question."""
#     if llm.model_name.startswith("gpt"):
#         docs = paperqa.Docs(llm=llm.model_name)
#     else:
#         docs = paperqa.Docs()  # uses default gpt model in paperqa

#     papers = paper_search(llm, query, path_registry)
#     if len(papers) == 0:
#         return "Failed. Not enough papers found"
#     not_loaded = 0
#     for path, data in papers.items():
#         try:
#             docs.add(path, data["citation"])
#         except (ValueError, FileNotFoundError, PdfReadError):
#             not_loaded += 1

#     print(
#         f"\nFound {len(papers)} papers"
#         + (f" but couldn't load {not_loaded}" if not_loaded > 0 else "")
#     )
#     answer = docs.query(query, k=k, max_sources=max_sources).formatted_answer
#     return "Succeeded. " + answer


def scholar2result_llm(llm, query, path_registry):
    paper_directory = path_registry.ckpt_papers
    print("Paper Directory", paper_directory)
    llm_name = llm.model_name
    if llm_name.startswith("gpt") or llm_name.startswith("claude"):
        settings = paperqa.Settings(
            llm=llm_name,
            summary_llm=llm_name,
            temperature=llm.temperature,
            paper_directory=paper_directory,
        )
    else:
        settings = paperqa.Settings(
            temperature=llm.temperature,  # uses default gpt model in paperqa
            paper_directory=paper_directory,
        )
    response = paperqa.ask(query, settings=settings)
    answer = response.answer.formatted_answer
    if "I cannot answer." in answer:
        answer += f" Check to ensure there's papers in {paper_directory}"
    print(answer)
    return answer


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = (
        "Useful to answer questions that may be found in literature. "
        "Ask a specific question as the input."
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
