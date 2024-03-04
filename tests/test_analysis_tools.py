import os
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import MDAnalysis as mda
import numpy as np
import pytest

from mdagent.tools.base_tools import VisFunctions
import mdagent.tools.base_tools.analysis_tools.ppi_tools
from mdagent.tools.base_tools.analysis_tools.plot_tools import PlottingTools
from mdagent.tools.base_tools.analysis_tools.ppi_tools import ppi_distance
from mdagent.utils import PathRegistry


@pytest.fixture
def get_registry():
    return PathRegistry()


@pytest.fixture
def plotting_tools(get_registry):
    return PlottingTools(get_registry)


@pytest.fixture
def vis_fxns(get_registry):
    return VisFunctions(get_registry)


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


def test_process_csv(plotting_tools):
    mock_csv_content = "Time,Value1,Value2\n1,10,20\n2,15,25"
    mock_reader = MagicMock()
    mock_reader.fieldnames = ["Time", "Value1", "Value2"]
    mock_reader.__iter__.return_value = iter(
        [
            {"Time": "1", "Value1": "10", "Value2": "20"},
            {"Time": "2", "Value1": "15", "Value2": "25"},
        ]
    )
    plotting_tools.file_path = "mock_file.csv"
    plotting_tools.file_name = "mock_file.csv"
    with patch("builtins.open", mock_open(read_data=mock_csv_content)):
        with patch("csv.DictReader", return_value=mock_reader):
            plotting_tools.process_csv()

    assert plotting_tools.headers == ["Time", "Value1", "Value2"]
    assert len(plotting_tools.matched_headers) == 1
    assert plotting_tools.matched_headers[0][1] == "Time"
    assert len(plotting_tools.data) == 2
    assert (
        plotting_tools.data[0]["Time"] == "1"
        and plotting_tools.data[0]["Value1"] == "10"
    )


def test_plot_data(plotting_tools):
    # Test successful plot generation
    data_success = [
        {"Time": "1", "Value1": "10", "Value2": "20"},
        {"Time": "2", "Value1": "15", "Value2": "25"},
    ]
    headers = ["Time", "Value1", "Value2"]
    matched_headers = [(0, "Time")]

    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
        "matplotlib.pyplot.xlabel"
    ), patch("matplotlib.pyplot.ylabel"), patch("matplotlib.pyplot.title"), patch(
        "matplotlib.pyplot.savefig"
    ), patch(
        "matplotlib.pyplot.close"
    ):
        plotting_tools.data = data_success
        plotting_tools.headers = headers
        plotting_tools.matched_headers = matched_headers
        created_plots = plotting_tools.plot_data()
        assert "time_vs_value1.png" in created_plots
        assert "time_vs_value2.png" in created_plots

    # Test failure due to non-numeric data
    data_failure = [
        {"Time": "1", "Value1": "A", "Value2": "B"},
        {"Time": "2", "Value1": "C", "Value2": "D"},
    ]

    plotting_tools.data = data_failure
    plotting_tools.headers = headers
    plotting_tools.matched_headers = matched_headers

    with pytest.raises(Exception) as excinfo:
        plotting_tools.plot_data()
        assert "All plots failed due to non-numeric data." in str(excinfo.value)


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(path_to_cif, vis_fxns):
    result = vis_fxns.run_molrender(path_to_cif)
    assert result == "Visualization created"


def test_find_png(vis_fxns):
    vis_fxns.starting_files = os.listdir(".")
    test_file = "test_image.png"
    with open(test_file, "w") as f:
        f.write("")
    png_files = vis_fxns._find_png()
    assert test_file in png_files

    os.remove(test_file)


def test_create_notebook(path_to_cif, vis_fxns):
    result = vis_fxns.create_notebook(path_to_cif)
    path_to_notebook = path_to_cif.split(".")[0] + "_vis.ipynb"
    os.remove(path_to_notebook)
    assert result == "Visualization Complete"
    

pdb_string = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  
ATOM      3  C   ALA A   1       1.458   1.527   0.000  1.00 20.00           C  
ATOM      4  O   ALA A   1       0.000   1.527   0.000  1.00 20.00           O  
ATOM      5  CB  ALA A   1       1.458  -0.500   1.500  1.00 20.00           C  
ATOM      6  N   GLY B   2      -1.458   0.000   0.000  1.00 20.00           N  
ATOM      7  CA  GLY B   2      -2.916   0.000   0.000  1.00 20.00           C  
ATOM      8  C   GLY B   2      -2.916   1.527   0.000  1.00 20.00           C  
ATOM      9  O   GLY B   2      -1.458   1.527   0.000  1.00 20.00           O  
ATOM     10  N   GLY B   3      -4.374   1.527   0.000  1.00 20.00           N  
TER
END
"""
pdb_file_like = StringIO(pdb_string)
u = mda.Universe(pdb_file_like, format="PDB")

@pytest.fixture
def mock_mda_universe():
    # pdb_string = """
    # ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  
    # ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  
    # ATOM      3  C   ALA A   1       1.458   1.527   0.000  1.00 20.00           C  
    # ATOM      4  O   ALA A   1       0.000   1.527   0.000  1.00 20.00           O  
    # ATOM      5  CB  ALA A   1       1.458  -0.500   1.500  1.00 20.00           C  
    # ATOM      6  N   GLY B   2      -1.458   0.000   0.000  1.00 20.00           N  
    # ATOM      7  CA  GLY B   2      -2.916   0.000   0.000  1.00 20.00           C  
    # ATOM      8  C   GLY B   2      -2.916   1.527   0.000  1.00 20.00           C  
    # ATOM      9  O   GLY B   2      -1.458   1.527   0.000  1.00 20.00           O  
    # ATOM     10  N   GLY B   3      -4.374   1.527   0.000  1.00 20.00           N  
    # TER
    # END
    # """
    # pdb_file_like = StringIO(pdb_string)
    # u = mda.Universe(pdb_file_like, format="PDB")
    with patch('mdagent.tools.base_tools.analysis_tools.ppi_tools.mda.Universe', return_value=u) as mock_universe:
        #mock_universe.return_value = u
        yield mock_universe

def test_ppi_distance(mock_mda_universe):
    file_path = "dummy_path.pdb"
    avg_dist = ppi_distance(file_path)
    assert avg_dist > 0, "Expected a positive average distance"

