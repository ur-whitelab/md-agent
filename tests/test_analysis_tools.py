import os
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import MDAnalysis as mda
import numpy as np
import pytest

from mdagent.tools.base_tools import VisFunctions
from mdagent.tools.base_tools.analysis_tools.inertia import (
    MomentOfInertia,
    calculate_moment_of_inertia,
    load_traj,
    save_to_csv,
)
from mdagent.tools.base_tools.analysis_tools.plot_tools import PlottingTools
from mdagent.tools.base_tools.analysis_tools.ppi_tools import ppi_distance
from mdagent.tools.base_tools.analysis_tools.rmsd_tools import RMSDFunctions
from mdagent.tools.base_tools.analysis_tools.sasa import (
    SASAAnalysis,
    SolventAccessibleSurfaceArea,
)


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
        assert "timevsvalue1" in created_plots
        assert "timevsvalue2" in created_plots

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
    (f"{vis_fxns.path_registry.ckpt_figures}/{path_to_cif.split('.')[0]}_vis.ipynb")
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

    message = rmsd_functions.compute_rmsd(selection="backbone", plot=False)

    mock_mda_universe.assert_called_once()
    mock_rmsd_run.assert_called_once()
    mock_savetxt.assert_called_once()
    args, kwargs = mock_savetxt.call_args
    assert args[0].startswith(
        f"{rmsd_functions.path_registry.ckpt_records}/{rmsd_functions.filename}_"
    ), "np.savetxt called with unexpected file name"
    # assert "test_rmsd.csv" in args, "Expected np.savetxt to save to correct file"
    assert "Average RMSD is 0.15" in message, "Expected correct average RMSD in message"
    assert "Final RMSD is 0.2" in message, "Expected correct final RMSD in message"
    assert "Saved to test_rmsd_" in message, "Expected correct save file message"


@pytest.mark.parametrize("plot_enabled", [True, False])
def test_compute_rmsd_plotting(
    plot_enabled, mock_mda_universe, mock_plt_savefig, rmsd_functions
):
    pdb_name = rmsd_functions.pdb_name
    rmsd_functions.filename = f"rmsd_{pdb_name}"
    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
        "matplotlib.pyplot.xlabel"
    ), patch("matplotlib.pyplot.ylabel"), patch("matplotlib.pyplot.title"), patch(
        "matplotlib.pyplot.close"
    ):
        message = rmsd_functions.compute_rmsd(selection="backbone", plot=plot_enabled)
    if plot_enabled:
        mock_plt_savefig.assert_called_once()
        args, _ = mock_plt_savefig.call_args
        assert args[0].startswith(
            f"{rmsd_functions.path_registry.ckpt_figures}/{rmsd_functions.filename}"
        ), "plt.savefig called with unexpected file path"
        assert (
            f"Plotted RMSD over time for {pdb_name}."
            f" Saved to {rmsd_functions.filename}" in message
        ), "Expected correct plotting message"
        assert f"Saved to {rmsd_functions.filename}" in message  # csv file
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


@pytest.mark.parametrize("plot", [True, False])
def test_process_rmsf_results(rmsd_functions, mock_plt_savefig, mock_savetxt, plot):
    rmsd_functions.filename = "test_rmsf"
    mock_atoms = MagicMock()
    mock_atoms.resids = np.arange(1, 11)
    mock_atoms.resnums = np.arange(1, 11)
    mock_rmsf_values = np.random.rand(10)
    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
        "matplotlib.pyplot.xlabel"
    ), patch("matplotlib.pyplot.ylabel"), patch("matplotlib.pyplot.title"), patch(
        "matplotlib.pyplot.close"
    ):
        message = rmsd_functions.process_rmsf_results(
            mock_atoms, mock_rmsf_values, plot=plot
        )
    mock_savetxt.assert_called_once()
    args, _ = mock_savetxt.call_args
    assert args[0].startswith(
        f"{rmsd_functions.path_registry.ckpt_records}/{rmsd_functions.filename}_"
    ), "CSV file path passed to np.savetxt doesn't match expected pattern."
    assert "Saved RMSF data to" in message, "Expected save message not found in return."
    if plot:
        mock_plt_savefig.assert_called_once()
        savefig_args, _ = mock_plt_savefig.call_args
        assert savefig_args[0].startswith(
            f"{rmsd_functions.path_registry.ckpt_figures}/FIG_{rmsd_functions.filename}_"
        ), "Plot file path passed to plt.savefig doesn't match expected pattern."
        assert "Plotted RMSF. Saved to" in message, "Expected plot message not found."

    else:
        mock_plt_savefig.assert_not_called()


################### SASA ######################


@pytest.fixture
def get_sasa_functions_with_files(get_registry):
    registry = get_registry("raw", True)
    traj_fileid = "rec0_butane_123456"
    top_fileid = "top_sim0_butane_123456"
    return SASAAnalysis(registry, top_fileid, traj_fileid)


def test_sasa_analysis_init_success(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        analysis = SASAAnalysis(
            registry, "top_sim0_butane_123456", "rec0_butane_123456"
        )
        assert mocked_get_mapped_path.call_count == 2
        assert analysis.path_registry == registry
        assert analysis.molecule_name == "sim0_butane_123456"
        assert analysis.traj is not None


def test_sasa_analysis_init_success_no_traj(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        analysis = SASAAnalysis(registry, "top_sim0_butane_123456", mol_name="butane")
        mocked_get_mapped_path.assert_called_once()
        assert analysis.path_registry == registry
        assert analysis.molecule_name == "butane"


def test_sasa_tool_init(get_registry):
    registry = get_registry("raw", False)
    tool = SolventAccessibleSurfaceArea(path_registry=registry)
    assert tool.name == "SolventAccessibleSurfaceArea"
    assert tool.path_registry == registry


@patch(
    "mdagent.tools.base_tools.analysis_tools.sasa.os.path.exists", return_value=False
)
@patch("mdagent.tools.base_tools.analysis_tools.sasa.np.savetxt")
def test_calculate_sasa(mock_savetxt, mock_exists, get_sasa_functions_with_files):
    analysis = get_sasa_functions_with_files
    result = analysis.calculate_sasa()
    assert "SASA values computed and saved" in result
    mock_savetxt.assert_called_once()
    assert analysis.sasa is not None
    assert analysis.residue_sasa is not None


@patch("mdagent.tools.base_tools.analysis_tools.sasa.plt.savefig")
@patch("mdagent.tools.base_tools.analysis_tools.sasa.plt.close")
def test_plot_sasa(mock_close, mock_savefig, get_sasa_functions_with_files):
    analysis = get_sasa_functions_with_files
    analysis.sasa = np.array([1, 2, 3])  # example data
    analysis.residue_sasa = np.array([[1, 2], [3, 4]])
    with patch.object(SASAAnalysis, "calculate_sasa") as mock_calc:
        result = analysis.plot_sasa()
        mock_calc.assert_not_called()
        mock_savefig.assert_called_once()
        assert "SASA analysis completed" in result


################### INERTIA ######################

# tests for  helper functions


def test_load_traj_only_topology(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        traj = load_traj(registry, "top_sim0_butane_123456")
        mocked_get_mapped_path.assert_called_once()
        assert traj is not None


def test_load_traj_topology_and_traj(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        traj = load_traj(registry, "top_sim0_butane_123456", "rec0_butane_123456")
        assert mocked_get_mapped_path.call_count == 2
        assert traj is not None


def test_load_traj_fail_top_fileid(get_registry):
    registry = get_registry("raw", False)
    with pytest.raises(ValueError) as exc:
        load_traj(registry, "top_invalid")
    assert "Topology File ID not found in path registry" in str(exc.value)


def test_load_traj_fail_traj_fileid(get_registry):
    registry = get_registry("raw", True)
    with pytest.raises(ValueError) as exc:
        load_traj(registry, "top_sim0_butane_123456", "traj_invalid")
    assert "Trajectory File ID not found in path registry" in str(exc.value)


def test_save_to_csv(get_registry):
    registry = get_registry("raw", False)
    data = np.array([[1, 2], [3, 4]])
    with patch("os.path.exists", return_value=False):
        with patch("numpy.savetxt", return_value=None):
            csv_path = save_to_csv(registry, data, "test_id", "Description of data")
            assert "test_id.csv" in csv_path


# testing core inertia functions


def test_moi_tool_init(get_registry):
    registry = get_registry("raw", False)
    tool = MomentOfInertia(path_registry=registry)
    assert tool.name == "MomentOfInertia"
    assert tool.path_registry == registry


def test_calculate_moment_of_inertia(get_registry):
    registry = get_registry("raw", True)
    top_fileid = "top_sim0_butane_123456"
    traj_fileid = "rec0_butane_123456"
    msg = calculate_moment_of_inertia(registry, top_fileid, traj_fileid)
    assert "Average Moment of Inertia Tensor:" in msg
    assert "saved to:" in msg
    assert "MOI_sim0_butane" in msg

    mol_name = "butane"
    msg = calculate_moment_of_inertia(registry, top_fileid, traj_fileid, mol_name)
    assert "MOI_butane" in msg
