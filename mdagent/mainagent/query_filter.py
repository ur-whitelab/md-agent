from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import outlines
from outlines import generate, models
from pydantic import BaseModel


@dataclass
class Parameters:
    Temperature: Optional[str]
    Pressure: Optional[str]
    Time: Optional[str]
    ForceField: Optional[str]
    WaterModel: Optional[str]
    SaltConcentration: Optional[str]
    pH: Optional[str]
    Solvate: Optional[bool]
    Ensemble: Optional[str]
    Other_Parameters: Optional[str]


class Task_type(str, Enum):
    question = "question"
    preprocessing = "preprocessing"
    simulation = "simulation"
    postnalysis = "postanalysis"


class FilteredQuery(BaseModel):
    Main_Task: str
    Subtask_types: List[Task_type]  # conlist(Task_type, min_length=1)
    ProteinS: str
    Parameters: Parameters
    UserProposedPlan: List[str]  # conlist(str, min_length=0]


@dataclass
class Example:
    Raw_query: str
    Filtered_Query: FilteredQuery


@outlines.prompt
def query_filter(raw_query, examples: list[Example]):
    """You are about to organize an user query. User will
    ask for a specific Molecular Dynamics related task, from wich you will
    extract:
    1. The main task of the query
    2. A list of subtasks that are part of the main task
    3. The protein of interest mentioned in the raw query (as a PDB ID,
      UniProt ID, name, or sequence)
    4. Parameters or conditions specified by the user for the simulation
    5. The plan proposed by the user for the simulation (if any)


    {% for example in examples %}
    Raw Query: "{{ example.Raw_query }}"
    RESULT: {
        "Main_Task": "{{ example.Filtered_Query.Main_Task }}",
        "Subtask_types": "{{ example.Filtered_Query.Subtask_types }}",
        "ProteinS": "{{ example.Filtered_Query.ProteinS }}",
        "Parameters": "{{ example.Filtered_Query.Parameters }}",
        "UserProposedPlan": "{{ example.Filtered_Query.UserProposedPlan }}"}
    {% endfor %}

    Here is the new raw query that you need to filter:
    Raw Query: {{raw_query}}
    RESULT:
    """


examples = [
    Example(
        Raw_query="I want a simulation of 1A3N at 280K",
        Filtered_Query=FilteredQuery(
            Main_Task="Simulate 1A3N at 280K",
            Subtask_types=["simulation"],
            ProteinS="1A3N",
            Parameters=Parameters(
                Temperature="280K",
                Pressure=None,
                Time=None,
                ForceField=None,
                WaterModel=None,
                SaltConcentration=None,
                pH=None,
                Solvate=None,
                Ensemble=None,
                Other_Parameters=None,
            ),
            UserProposedPlan=[],
        ),
    ),
    Example(
        Raw_query="What is the best force field for 1A3N?",
        Filtered_Query=FilteredQuery(
            Main_Task="Answer the question: best force field for 1A3N?",
            Subtask_types=["question"],
            ProteinS="1A3N",
            Parameters=Parameters(
                Temperature=None,
                Pressure=None,
                Time=None,
                ForceField=None,
                WaterModel=None,
                SaltConcentration=None,
                pH=None,
                Solvate=None,
                Ensemble=None,
                Other_Parameters=None,
            ),
            UserProposedPlan=[],
        ),
    ),
    Example(
        Raw_query="""Calculate the Radial Distribution Function of 1A3N with
            water. Youll have to download the PDB file, clean it, and solvate it
            for the simulation. The trajectory and
            topology files can be used to calculate the RDF.""",
        Filtered_Query=FilteredQuery(
            Main_Task="Calculate the Radial Distribution Function of 1A3N with water.",
            Subtask_types=["preprocessing", "simulation", "postanalysis"],
            ProteinS="1A3N",
            Parameters=Parameters(
                Temperature=None,
                Pressure=None,
                Time=None,
                ForceField=None,
                WaterModel=None,
                SaltConcentration=None,
                pH=None,
                Solvate=True,
                Ensemble=None,
                Other_Parameters=None,
            ),
            UserProposedPlan=[
                "Downlaod PDB file for 1A3N",
                "Clean/Pre-process the PDB file",
                "Calculate the Radial Distribution Function with water.",
                "With the trajectory and topology files, calculate the RDF.",
            ],
        ),
    ),
]


def create_filtered_query(raw_query, model="gpt-3.5-turbo", examples=examples):
    filter_model = models.openai(model)
    generator = generate.text(filter_model)
    return generator(query_filter(raw_query, examples=examples))
