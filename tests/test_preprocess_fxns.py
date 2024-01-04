import os

import pytest

from mdagent.tools.base_tools.preprocess_tools import CleaningTools, get_pdb
from mdagent.utils import PathRegistry

# Test functions in preprocess tools


@pytest.fixture
def get_registry():
    return PathRegistry()


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


# test get pdb file
@pytest.fixture
def fibronectin():
    return "fibronectin pdb"


def test_getpdb(fibronectin, get_registry):
    name = get_pdb(fibronectin, get_registry)
    assert name.endswith(".pdb")


# test cleaning functions
@pytest.fixture
def cleaning_fxns():
    return CleaningTools()


def test_add_hydrogens_and_remove_water(path_to_cif, cleaning_fxns, get_registry):
    result = cleaning_fxns._add_hydrogens_and_remove_water(path_to_cif, get_registry)
    assert "Cleaned File" in result  # just want to make sur the function ran
