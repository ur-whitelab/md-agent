import os

import pytest

from mdagent.tools.vis_tools import VisFunctions


@pytest.fixture
def path_to_cif():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(tests_dir, "3pqr.cif")
    return doc_path


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
