from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import outlines
from outlines import generate, models
from pydantic import BaseModel

################################################################

"""
The following is based on outlines documentations and examples
https://outlines-dev.github.io/outlines/welcome/

Classes and functions described in this file:
- Parameters
- Task_type
- FilteredQuery
- Example
- query_filter
- create_filtered_query


query_filter: A function that takes a raw query and a list of examples (Example classes)
and returns a "prompt for filtering" that include the examples (similar to
Few-shot prompting).

create_filtered_query: A function that uses (so far an openai model) and the
query_filter function to create a filtered query

Parameters: A dataclass that represents the parameters of a molecular dynamics. I've
included Temperature, Pressure, Time, ForceField, WaterModel, SaltConcentration, pH,
Solvate, Ensemble, and Other_Parameters. All parameters are optional, depending on what
the user includes in its input.
it has two main methods: 1) to parse a string into a Parameters object.
and a __repr__ method to print the object as a string (useful for making a
pretty prompt)


Task_type: An Enum class that represents the different types of tasks that a user can
ask for, or that the model used in create_filtered_query assumes it needs.
 It has a __repr__ method to print the object as a string (useful for making a
pretty prompt) and a parse_task_type_string method to parse a string into a
Task_type object.

FilteredQuery: A pydantic BaseModel class that represents the final structure that will
summarize the info from the users request. It uses the two classes defined above.

Example: A dataclass that represents an example of a raw query and its filtered query.
"""


################################################################
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

    def __repr__(self) -> str:
        return (
            f"Parameters(Temperature = '{self.Temperature}',"
            f"Pressure = '{self.Pressure}', "
            f"Time = '{self.Time}',"
            f"ForceField = '{self.ForceField}',"
            f"WaterModel = '{self.WaterModel}', "
            f"SaltConcentration = '{self.SaltConcentration}',"
            f"pH = '{self.pH}',"
            f"Solvate = '{self.Solvate}', "
            f"Ensemble = '{self.Ensemble}', "
            f"Other_Parameters = '{self.Other_Parameters}'"
            ")"
        )

    @staticmethod
    def parse_parameters_string(param_str):
        # Remove the 'Parameters' prefix and parentheses
        param_str = param_str.replace("Parameters(", "").replace(")", "")
        # Split the string into key-value pairs
        pairs = param_str.split(",")
        param_dict = {}
        for pair in pairs:
            key, value = pair.split("=")
            key = key.strip()
            value = value.strip().strip("'")
            # Convert 'True' and 'False' strings to boolean values
            if value == "True":
                value = True
            elif value == "False":
                value = False
            param_dict[key] = value
        return Parameters(**param_dict)

    def items(self):
        return self.__dict__.items()


class Task_type(str, Enum):
    question = "Question"
    preprocessing = "Preprocessing"
    preparation = "Preparation"
    simulation = "Simulation"
    postnalysis = "Postanalysis"

    def __repr__(self):
        return f"{self.value}"

    @staticmethod
    def parse_task_type_string(task_type_str):
        if type(task_type_str) == str:
            if task_type_str.startswith("["):
                task_type_str = task_type_str.replace("[", "").replace("]", "")
                task_type_str = task_type_str.split(",")
                return [Task_type(task_type.strip()) for task_type in task_type_str]
            return Task_type(task_type_str.strip())
        elif type(task_type_str) == list:
            return [Task_type(task_type) for task_type in task_type_str]
        elif type(task_type_str) == Task_type:
            return task_type_str

    def __str__(self):
        return self.value


class FilteredQuery(BaseModel):
    Main_Task: str
    Subtask_types: List[Task_type]  # conlist(Task_type, min_length=1)
    ProteinS: List[str]
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
            Subtask_types=["Simulation"],
            ProteinS=["1A3N"],
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
            Subtask_types=["Question"],
            ProteinS=["1A3N"],
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
            Subtask_types=["Preprocessing", "Simulation", "Postanalysis"],
            ProteinS=["1A3N"],
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
