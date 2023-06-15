import os

import pytest

from mdagent.general_tools import create_notebook, dummy_function, run_molrender


@pytest.fixture
def path_to_cif():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "3pqr.cif")
    return doc_path


# dummy test, remove when we have real tests here
def test_dummy():
    dummy_output = dummy_function()
    assert dummy_output == 46


def test_dummy_run():
    assert True


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(path_to_cif):
    result = run_molrender(path_to_cif)
    assert result == "Visualization created"


def test_create_notebook(path_to_cif):
    result = create_notebook(path_to_cif)
    assert result == "Visualization Complete"
