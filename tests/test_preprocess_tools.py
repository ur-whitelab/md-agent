import os

import pytest

from mdagent.tools.base_tools import CleaningTools
from mdagent.utils import PathRegistry


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


@pytest.fixture
def get_registry():
    return PathRegistry()


@pytest.fixture
def cleaning_fxns(get_registry):
    return CleaningTools(get_registry)


def test_add_hydrogens_and_remove_water(path_to_cif, cleaning_fxns):
    result = cleaning_fxns._add_hydrogens_and_remove_water(path_to_cif)
    path_to_cleaned_file = "tidy_" + path_to_cif
    os.remove(path_to_cleaned_file)
    assert "Cleaned File" in result


def test_standard_cleaning(path_to_cif, cleaning_fxns):
    result = cleaning_fxns._standard_cleaning(path_to_cif)
    path_to_cleaned_file = "tidy_" + path_to_cif
    os.remove(path_to_cleaned_file)
    assert "Cleaned File" in result
