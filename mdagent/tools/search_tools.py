import json

import langchain
import pqapi
from langchain import LLMChain, PromptTemplate
from langchain.tools import BaseTool


def Prompt_summary(query: str):
    llm = langchain.chat_models.ChatOpenAI(
        temperature=0.05, model_name="gpt-4", request_timeout=2000, max_tokens=2000
    )

    prompt_template = """Your input is the original query. Your
                        task is to parse through the user query.
                        and provide a summary of the file path input,
                        the type of preprocessing needed (this is the
                        same as cleaning the file), the forcefield
                        used for the simulation,
                        the ensemble of the simulation, the integrator needed,
                        the number of steps, the timestep, the temperature,
                        and other instructions.
                        and follow the format "name: description.

                        File Path: what is the file path of the file
                        you are using? it must include a .cif or .pdb extension.
                        Preprocessing: what preprocessing is needed?
                        you can choose from the following: standard cleaning,
                        remove water, add hydrogens, add hydrogens and remove
                        water. The default is add hydrogens and remove water.
                        Forcefield: what forcefields are you using?
                        you can choose from the following: AMBER, CHARMM,
                        OPLS, GROMACS. The default is "amber14-all.xml, tip3p.xml".
                        Ensemble: what ensemble are you using?
                        you can choose from the following:
                        NPT, NVT, NVE. This default is NVT
                        Integrator: what integrator are you using?
                        you can choose from the following:
                        Langevin, Verlet, Brownian.
                        The default depends on the ensemble
                        (NPT -> Langevin, NVT -> Langevin, NVE -> Verlet).
                        Number of Steps: how many steps
                        are you using? The default is 1000.
                        Timestep: what is the timestep?
                        The default is 1 fs.
                        Temperature: what is the temperature?
                        The default is 300 K.
                        Friction: what is the friction coefficient?
                        The default is 1.0 (1/ps)
                        Other Instructions: what other instructions do you have?
                        The default is none.

                        If there is not enough information in a category,
                        you may fill in with the default, but explicitly state so.
                        Here is the information:{query}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.run(" ".join(query))


def _save_to_file(summary: str, filename: str):
    """Parse the summary string and
    save it to a file in JSON format."""
    # Split the summary into lines
    lines = summary.strip().split("\n")

    # Parse each line into a key and a value
    summary_dict = {}
    for line in lines:
        key, value = line.split(":")
        summary_dict[key.strip()] = value.strip()

    # Save the dictionary to a file
    with open(filename, "w") as f:
        json.dump(summary_dict, f)


class Scholar2ResultLLM(BaseTool):
    name = "LiteratureSearch"
    description = """Input a specific question,
                returns an answer from literature search."""

    pqa_key: str = ""

    def __init__(self, pqa_key: str):
        super().__init__()
        self.pqa_key = pqa_key

    def _run(self, question: str) -> str:
        """Use the tool"""
        response = pqapi.agent_query("default", question)
        return response.answer

    async def _arun(self, question: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError


class InstructionSummary(BaseTool):
    name = "Instruction Summary"
    description = """This tool will summarize the instructions
    given by the human. This is the first tool you will use.
    Input: Instructions or original query.
    Output: Summary of instructions"""

    def _run(self, query: str) -> str:
        summary = Prompt_summary(query)
        _save_to_file(summary, "simmulation_parameters.json")
        return summary

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
