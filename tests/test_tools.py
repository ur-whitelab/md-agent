import os
import warnings
from unittest.mock import MagicMock, mock_open, patch
from langchain.chat_models import ChatOpenAI
import pytest
from mdagent.tools.base_tools import Scholar2ResultLLM
from mdagent.tools.base_tools import VisFunctions, get_pdb
from mdagent.tools.base_tools.analysis_tools.plot_tools import plot_data, process_csv
from mdagent.utils import PathRegistry

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


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


# Test visualization tools
@pytest.fixture
def vis_fxns():
    return VisFunctions()


# Test MD utility tools
@pytest.fixture
def fibronectin():
    return "fibronectin pdb"


@pytest.fixture
def get_registry():
    return PathRegistry()


def test_process_csv():
    mock_csv_content = "Time,Value1,Value2\n1,10,20\n2,15,25"
    mock_reader = MagicMock()
    mock_reader.fieldnames = ["Time", "Value1", "Value2"]
    mock_reader.__iter__.return_value = iter(
        [
            {"Time": "1", "Value1": "10", "Value2": "20"},
            {"Time": "2", "Value1": "15", "Value2": "25"},
        ]
    )

    with patch("builtins.open", mock_open(read_data=mock_csv_content)):
        with patch("csv.DictReader", return_value=mock_reader):
            data, headers, matched_headers = process_csv("mock_file.csv")

    assert headers == ["Time", "Value1", "Value2"]
    assert len(matched_headers) == 1
    assert matched_headers[0][1] == "Time"
    assert len(data) == 2
    assert data[0]["Time"] == "1" and data[0]["Value1"] == "10"


def test_plot_data():
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
        created_plots = plot_data(data_success, headers, matched_headers)
        assert "time_vs_value1.png" in created_plots
        assert "time_vs_value2.png" in created_plots

    # Test failure due to non-numeric data
    data_failure = [
        {"Time": "1", "Value1": "A", "Value2": "B"},
        {"Time": "2", "Value1": "C", "Value2": "D"},
    ]

    with pytest.raises(Exception) as excinfo:
        plot_data(data_failure, headers, matched_headers)
        assert "All plots failed due to non-numeric data." in str(excinfo.value)


@pytest.mark.skip(reason="molrender is not pip installable")
def test_run_molrender(path_to_cif, vis_fxns):
    result = vis_fxns.run_molrender(path_to_cif)
    assert result == "Visualization created"


def test_create_notebook(path_to_cif, vis_fxns, get_registry):
    result = vis_fxns.create_notebook(path_to_cif, get_registry)
    assert result == "Visualization Complete"


def test_getpdb(fibronectin, get_registry):
    name, _ = get_pdb(fibronectin, get_registry)
    assert name.endswith(".pdb")

@pytest.fixture
def questions():
    qs = [
        "What are the effects of norhalichondrin B in mammals?",
    ]
    return qs[0]

@pytest.mark.skip(reason="This requires an API call")
def test_litsearch(questions):
    llm = ChatOpenAI()

    searchtool = Scholar2ResultLLM(llm=llm)
    for q in questions:
        ans = searchtool._run(q)
        assert isinstance(ans, str)
        assert len(ans) > 0
    #then if query folder exists one step back, delete it
    if os.path.exists("../query"):
        os.rmdir("../query")