import os
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import MDAnalysis as mda
import numpy as np
import pytest

from mdagent.tools.base_tools import VisFunctions
from mdagent.tools.base_tools.analysis_tools.plot_tools import PlottingTools
from mdagent.tools.base_tools.analysis_tools.ppi_tools import ppi_distance
from mdagent.tools.base_tools.analysis_tools.rmsd_tools import RMSDFunctions


@pytest.fixture
def plotting_tools(get_registry):
    return PlottingTools(get_registry("raw", False))


@pytest.fixture
def vis_fxns(get_registry):
    return VisFunctions(get_registry("raw", False))


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
        assert "FIG_timevsvalue1" in created_plots
        assert "FIG_timevsvalue2" in created_plots

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


@pytest.fixture
def rmsd_functions(get_registry):
    reg = get_registry("raw", True)
    pdb_file = reg.get_mapped_path("ALA_123456")
    rmsd_functions = RMSDFunctions(reg, pdb_file, "trajectory.dcd")
    rmsd_functions.trajectory = "trajectory.dcd"
    rmsd_functions.pdb_file = pdb_file
    return rmsd_functions


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
def mock_mda_universe(get_registry):
    # reg = get_registry("raw", True)
    # pdb_freg.get_mapped_path("ALA_123456")
    with patch(
        "mdagent.tools.base_tools.analysis_tools.ppi_tools.mda.Universe", return_value=u
    ) as mock_universe:
        yield mock_universe


@pytest.fixture
def mock_rmsd_run():
    mock_rmsd_results = MagicMock()
    mock_rmsd_results.rmsd = np.array([[0, 0.0, 0.1], [1, 10.0, 0.2]])
    with patch(
        "mdagent.tools.base_tools.analysis_tools.ppi_tools.mda.analysis.rms.RMSD",
        return_value=MagicMock(results=mock_rmsd_results),
    ) as mock:
        yield mock


@pytest.fixture
def mock_savetxt():
    with patch(
        "mdagent.tools.base_tools.analysis_tools.rmsd_tools.np.savetxt"
    ) as mock_savetxt:
        yield mock_savetxt


@pytest.fixture
def mock_plt_savefig():
    with patch("mdagent.tools.base_tools.analysis_tools.rmsd_tools.plt.savefig") as plt:
        yield plt


def test_ppi_distance(mock_mda_universe):
    file_path = "dummy_path.pdb"
    avg_dist = ppi_distance(file_path)
    assert avg_dist > 0, "Expected a positive average distance"


def test_calculate_rmsd(rmsd_functions):
    # Mock all related compute_* methods in rmsd_functions
    with patch.object(
        rmsd_functions, "compute_rmsd_2sets"
    ) as mock_compute_2sets, patch.object(
        rmsd_functions, "compute_rmsd"
    ) as mock_compute_rmsd, patch.object(
        rmsd_functions, "compute_2d_rmsd"
    ) as mock_compute_2d_rmsd, patch.object(
        rmsd_functions, "compute_rmsf"
    ) as mock_compute_rmsf:
        # Test rmsd_type="rmsd" with a reference file (call compute_rmsd_2sets)
        rmsd_functions.ref_file = "ref.pdb"
        rmsd_functions.calculate_rmsd(rmsd_type="rmsd")
        mock_compute_2sets.assert_called_once_with(selection="backbone")
        mock_compute_rmsd.assert_not_called()
        mock_compute_2d_rmsd.assert_not_called()
        mock_compute_rmsf.assert_not_called()

        mock_compute_2sets.reset_mock()

        # Test rmsd_type="rmsd" without a reference file (compute_rmsd should be called)
        rmsd_functions.ref_file = None
        rmsd_functions.calculate_rmsd(rmsd_type="rmsd")
        mock_compute_rmsd.assert_called_once_with(selection="backbone", plot=True)
        mock_compute_2sets.assert_not_called()
        mock_compute_2d_rmsd.assert_not_called()
        mock_compute_rmsf.assert_not_called()

        mock_compute_rmsd.reset_mock()

        # Test rmsd_type="pairwise_rmsd" (compute_2d_rmsd should be called)
        rmsd_functions.calculate_rmsd(rmsd_type="pairwise_rmsd")
        mock_compute_2d_rmsd.assert_called_once_with(
            selection="backbone", plot_heatmap=True
        )
        mock_compute_2sets.assert_not_called()
        mock_compute_rmsd.assert_not_called()
        mock_compute_rmsf.assert_not_called()

        mock_compute_2d_rmsd.reset_mock()

        # Test rmsd_type="rmsf" (compute_rmsf should be called)
        rmsd_functions.calculate_rmsd(rmsd_type="rmsf")
        mock_compute_rmsf.assert_called_once_with(selection="backbone", plot=True)
        mock_compute_2sets.assert_not_called()
        mock_compute_rmsd.assert_not_called()
        mock_compute_2d_rmsd.assert_not_called()

        # Test for invalid rmsd_type (should raise ValueError)
        with pytest.raises(ValueError):
            rmsd_functions.calculate_rmsd(rmsd_type="invalid_rmsd_type")


def test_compute_rmsd_2sets(mock_mda_universe, rmsd_functions):
    with patch(
        "mdagent.tools.base_tools.analysis_tools.ppi_tools.mda.analysis.rms.rmsd",
        return_value=0.5,
    ) as mock_rmsd:
        result = rmsd_functions.compute_rmsd_2sets(selection="backbone")
        assert "0.5" in result, "RMSD value should be present in the result string"
        mock_mda_universe.assert_called()
        mock_rmsd.assert_called()


def test_compute_rmsd(mock_mda_universe, mock_rmsd_run, mock_savetxt, rmsd_functions):
    rmsd_functions.filename = "test_rmsd"
    message = rmsd_functions.compute_rmsd(selection="backbone", plot=True)

    mock_mda_universe.assert_called()
    mock_rmsd_run.assert_called()
    mock_savetxt.assert_called()
    args, kwargs = mock_savetxt.call_args
    assert "test_rmsd.csv" in args, "Expected np.savetxt to save to correct file"
    assert "Average RMSD is 0.15" in message, "Expected correct average RMSD in message"
    assert "Final RMSD is 0.2" in message, "Expected correct final RMSD in message"
    assert "Saved to test_rmsd.csv" in message, "Expected correct save file message"


@pytest.mark.parametrize("plot_enabled", [True, False])
def test_compute_rmsd_plotting(
    plot_enabled, mock_mda_universe, mock_plt_savefig, rmsd_functions
):
    pdb_name = rmsd_functions.pdb_name
    rmsd_functions.filename = f"rmsd_{pdb_name}"
    message = rmsd_functions.compute_rmsd(selection="backbone", plot=plot_enabled)
    if plot_enabled:
        mock_plt_savefig.assert_called_once()
        assert (
            f"Plotted RMSD over time for {pdb_name}. Saved with plot id" in message
        ), "Expected correct plotting message"
        assert f"Saved to {rmsd_functions.filename}.csv" in message
    else:
        mock_plt_savefig.assert_not_called()


def test_compute_2d_rmsd(mock_mda_universe, mock_savetxt, rmsd_functions):
    rmsd_functions.filename = "test_pairwise_rmsd"
    patch_path = (
        "mdagent.tools.base_tools.analysis_tools.ppi_tools.mda.analysis."
        "diffusionmap.DistanceMatrix.run"
    )
    with patch(patch_path) as mock_distance_matrix_run:
        result = rmsd_functions.compute_2d_rmsd(
            selection="backbone", plot_heatmap=False
        )
        mock_mda_universe.assert_called()
        mock_distance_matrix_run.assert_called()
        mock_savetxt.assert_called()
        assert "Saved pairwise RMSD matrix" in result


@pytest.mark.parametrize("plot", [True, False])
def test_process_rmsf_results(
    rmsd_functions, tmp_path, mock_plt_savefig, mock_savetxt, plot
):
    mock_atoms = MagicMock()
    mock_atoms.resids = np.arange(1, 11)
    mock_atoms.resnums = np.arange(1, 11)
    mock_rmsf_values = np.random.rand(10)
    output_csv = tmp_path / "output_rmsf.csv"
    output_png = tmp_path / "output_rmsf.png"
    rmsd_functions.filename = str(tmp_path / "output_rmsf")
    message = rmsd_functions.process_rmsf_results(
        mock_atoms, mock_rmsf_values, plot=plot
    )
    mock_savetxt.assert_called_once()
    args, _ = mock_savetxt.call_args
    assert args[0] == str(output_csv), "CSV file path passed to np.savetxt don't match."

    if plot:
        mock_plt_savefig.assert_called_once_with(str(output_png))
        assert "Plotted RMSF. Saved to" in message
    else:
        mock_plt_savefig.assert_not_called()
    assert "Saved RMSF data to" in message


@pytest.mark.parametrize("plot", [True, False])
def test_compute_rmsf(rmsd_functions, mock_mda_universe, plot):
    with patch.object(
        rmsd_functions, "process_rmsf_results"
    ) as mocked_process_rmsf_results:
        mocked_process_rmsf_results.return_value = None
        rmsd_functions.compute_rmsf(selection="backbone", plot=plot)
        mock_mda_universe.assert_called()
        mocked_process_rmsf_results.assert_called_once()
        args, kwargs = mocked_process_rmsf_results.call_args
        selection = kwargs["selection"]
        plot_arg = kwargs["plot"]
        assert selection == "backbone"
        assert plot_arg is plot
