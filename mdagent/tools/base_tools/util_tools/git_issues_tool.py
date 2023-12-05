from typing import List, Optional

import requests
import tiktoken
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from serpapi import GoogleSearch

from mdagent.utils import _make_llm


class GitToolFunctions:
    """Class to store the functions of the tool."""

    """chain that can be used the tools for summarization or classification"""
    llm_ = _make_llm(
        model="gpt-3.5-turbo-16k", temp=0.05, verbose=False, max_tokens=2500
    )

    def _prompt_summary(self, query: str, output: str, llm: BaseLanguageModel = llm_):
        prompt_template = """You're receiving the following github issues and comments.
                            They come after looking for issues
                            in the openmm repo for the query: {query}.
                            The responses have the following format:
                            Issue: body of the issue
                            Comment: comments in response to the issue.
                            There are up to 5 comments per issue.
                            Some of the comments do not address the issue.
                            You job is to decide:
                            1) if the issue is relevant to the query.
                            2) if the comments are relevant to the issue.
                            Then, make a summary of the issue and comments.
                            Only keeping the relevant information.
                            If there are PDB files shared,
                            just add a few lines from them, not all of it.
                            If a comment is not relevant,
                            do not include it in the summary.
                            And if the issue is not relevant,
                            do not include it in the summary.
                            Keep in the summary all possible solutions given
                            in the comments if they are appropiate.
                            The summary should have at most 2.5k tokens.
                            The answer you have to summarize is:
                            {output}

                            you:"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["query", "output"]
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        return llm_chain.run({"query": query, "output": output})

    """Function to get the number of requests remaining for the Github API """

    def get_requests_remaining(self):
        url = "https://api.github.com/rate_limit"
        response = requests.get(url)
        return response.json()["rate"]["remaining"]

    def make_encoding(self):
        return tiktoken.encoding_for_model("gpt-4")


class SerpGitTool(BaseTool):
    name = "Openmm_Github_Issues_Search"
    description = """ Tool that searches inside
                    github issues in openmm. Make
                    your query as if you were googling something.
                    Input: Trying to run a simulation with a
                    custom forcefield error: error_code.
                    Output: Relevant issues with your query.
                    Input: """
    serp_key: Optional[str]

    def __init__(self, serp_key):
        super().__init__()
        self.serp_key = serp_key

    def _run(self, query: str):
        fxns = GitToolFunctions()
        # print("this is the key", self.serp_key)
        params = {
            "engine": "google",
            "q": "site:github.com/openmm/openmm/issues " + query,
            "api_key": self.serp_key,
        }
        encoding = fxns.make_encoding()
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results")
        if organic_results is None:
            if results.get("error"):
                return "error: " + results.get("error")
            else:
                return "Error: No 'organic_results' found"
        issues_numbers: List = (
            []
        )  # list that will contain issue id numbers retrieved from the google search
        number_of_results = (
            3  # number of results to be retrieved from the google search
        )
        print(len(organic_results), "results found with SERP API")
        for result in organic_results:
            if (
                len(issues_numbers) == number_of_results
            ):  # break if we have enough results
                break
            link = result["link"]
            number = int(link.split("/")[-1])
            # check if number is integer
            if isinstance(number, int):
                issues_numbers.append(number)

        # search for issues

        number_of_requests = len(issues_numbers) * 2  # 1 for comments, 1 for issues
        remaining_requests = fxns.get_requests_remaining()
        print("remaining requests", remaining_requests)
        if remaining_requests > number_of_requests:
            issues_dict = {}
            print("number of issues", len(issues_numbers))
            for number in issues_numbers:
                url_comments = f"https://api.github.com/repos/openmm/openmm/issues/{number}/comments"
                url_issues = (
                    f"https://api.github.com/repos/openmm/openmm/issues/{number}"
                )
                response_issues = requests.get(url_issues)
                response_comments = requests.get(url_comments)

                if (
                    response_issues.status_code == 200
                    and response_comments.status_code == 200
                ):
                    issues = response_issues.json()
                    issue = issues["title"]
                    body = issues["body"]
                    comments = response_comments.json()
                    body += f"\n\n Comments for issue {number}: \n"
                    for i, comment in enumerate(comments):
                        body += f"Answer#{i}:{comment['body']} \n"
                        if i > 5:  # up to 5 comments per issue should be enough,
                            # some issues have more than 100 comments
                            break  # TODO definitely summarize comments
                            # if there are more than x amount of comments.
                    issues_dict[f"{number}"] = [issue, body]
                else:
                    print(f"Error: {response_comments.status_code} for issue {number}")
                    continue

                # prepare the output
                output = ""
                for key in issues_dict.keys():
                    output += f"Issue {key}: {issues_dict[key][0]} \n"
                    output += f"Body: {issues_dict[key][1]} \n"

            num_tokens = len(encoding.encode(str(output)))
            if num_tokens > 4000:
                # summarize output
                output = fxns._prompt_summary(query, output)
            return output
        else:
            return "Not enough requests remaining for Github API. Try again later"

    def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Name2PDB does not support async")
