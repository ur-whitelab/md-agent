# test_query_filter.py
import pytest

try:
    from mdagent.agent.query_filter import (
        Example,
        FilteredQuery,
        Parameters,
        query_filter,
    )
except Exception:
    pytest.skip(allow_module_level=True)


@pytest.mark.skip(reason="We skip because query filter was moved to WIP branch.")
def test_parameters_repr():
    params = Parameters(
        Temperature=300,
        Pressure=1.0,
        Time=None,
        ForceField="AMBER",
        WaterModel=None,
        SaltConcentration=None,
        pH=7.0,
        Solvate=None,
        Ensemble="NVT",
        Other_Parameters=None,
    )
    expected_repr = (
        "Parameters(Temperature = '300',"
        "Pressure = '1.0',"
        "Time = 'None',"
        "ForceField = 'AMBER',"
        "WaterModel = 'None',"
        "SaltConcentration = 'None',"
        "pH = '7.0',"
        "Solvate = 'None',"
        "Ensemble = 'NVT',"
        "Other_Parameters = 'None')"
    )
    assert repr(params) == expected_repr


@pytest.mark.skip(reason="We skip because query filter was moved to WIP branch.")
def test_query_filter_simple():
    raw_query = "Find the melting point of NaCl"
    example = Example(
        Raw_query="Find the melting point of NaCl",
        Filtered_Query=FilteredQuery(
            Main_Task="Find the melting point of NaCl",
            Subtask_types=["Question"],
            ProteinS=["None"],
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
    )
    examples = [example]
    expected_output = """You are about to organize an user query. User will
ask for a specific Molecular Dynamics related task, from wich you will
extract:
1. The main task of the query
2. A list of subtasks that are part of the main task
3. The protein of interest mentioned in the raw query (as a PDB ID,
  UniProt ID, name, or sequence)
4. Parameters or conditions specified by the user for the simulation
5. The plan proposed by the user for the simulation (if any)


Raw Query: "Find the melting point of NaCl"
RESULT: {
    "Main_Task": "Find the melting point of NaCl",
    "Subtask_types": "[Question]",
    "ProteinS": "['None']",
    "Parameters": "Parameters(Temperature = 'None',\
Pressure = 'None',\
Time = 'None',\
ForceField = 'None',\
WaterModel = 'None',\
SaltConcentration = 'None',\
pH = 'None',\
Solvate = 'None',\
Ensemble = 'None',\
Other_Parameters = 'None')",
    "UserProposedPlan": "[]"}

Here is the new raw query that you need to filter:
Raw Query: Find the melting point of NaCl
RESULT:"""
    # Assuming the function query_filter returns a string for simplicity
    assert query_filter(raw_query, examples=examples) == expected_output
