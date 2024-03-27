import os

import pytest

from mdagent.tools.base_tools import CleaningToolFunction


@pytest.fixture
def path_to_cif():
    # Save original working directory
    original_cwd = os.getcwd()

    # Change current working directory to the directory where the CIF file is located
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tests_dir)

    # Yield the filename only
    filename_only = "3pqr.cif"
    yield filename_only

    # Restore original working directory after the test is done
    os.chdir(original_cwd)


def test_cleaning_function(get_registry):
    reg = get_registry("raw", True)
    tool = CleaningToolFunction(path_registry=reg)
    assert tool.path_registry
    assert tool.name == "CleaningToolFunction"
    assert tool.path_registry == reg
    prompt = {
        "pdb_id": "ALA_123456",
        "replace_nonstandard_residues": True,
        "add_missing_atoms": True,
        "remove_heterogens": True,
        "remove_water": True,
        "add_hydrogens": True,
        "add_hydrogens_ph": 7.0,
    }
    result = tool._run(**prompt)
    assert "File cleaned" in result
