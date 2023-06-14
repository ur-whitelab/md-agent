import os

import pytest

from mdagent.tools.md_util_tools import get_pdb
from mdagent.tools.vis_tools import VisFunctions

# from mdagent.general_tools import dummy_function, name2pdb


@pytest.fixture
def path_to_cif():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "3pqr.cif")
    return doc_path


# Test visualization tools
@pytest.fixture
def vis_fxns():
    return VisFunctions()


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(
    path_to_ci,
):
    result = vis_fxns.run_molrender(path_to_cif, vis_fxns)
    assert result == "Visualization created"


def test_create_notebook(path_to_cif, vis_fxns):
    result = vis_fxns.create_notebook(path_to_cif)
    assert result == "Visualization Complete"


# Test MD utility tools
@pytest.fixture
def fibronectin():
    return "fibronectin"


# test name2pdb function
def test_getpdb(fibronectin):
    name = get_pdb(fibronectin)
    assert name == "1X3D.cif"
