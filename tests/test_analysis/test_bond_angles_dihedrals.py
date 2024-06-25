from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool import (
    ComputeAngles,
    ComputeChi1,
    ComputeChi2,
    ComputeChi3,
    ComputeChi4,
    ComputeDihedrals,
    ComputeOmega,
    ComputePhi,
    ComputePsi,
    RamachandranPlot,
)


# Fixture to patch 'load_single_traj'
@pytest.fixture
def patched_load_single_traj():
    with patch(
        "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
    ) as mock_load_single_traj:
        yield mock_load_single_traj


@pytest.fixture
def compute_angles_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputeAngles(path_registry)


@pytest.fixture
def compute_dihedrals_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputeDihedrals(path_registry)


@pytest.fixture
def compute_phi_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputePhi(path_registry)


@pytest.fixture
def compute_psi_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputePsi(path_registry)


@pytest.fixture
def compute_chi1_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputeChi1(path_registry)


@pytest.fixture
def compute_chi2_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputeChi2(path_registry)


@pytest.fixture
def compute_chi3_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputeChi3(path_registry)


@pytest.fixture
def compute_chi4_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputeChi4(path_registry)


@pytest.fixture
def compute_omega_tool(get_registry):
    path_registry = get_registry("raw", True)
    return ComputeOmega(path_registry)


@pytest.fixture
def ramachandran_plot(get_registry):
    path_registry = get_registry("raw", True)
    return RamachandranPlot(path_registry)


@patch("mdtraj.compute_angles")
@patch("matplotlib.pyplot.savefig")
def test_run_success_compute_angles(
    mock_savefig, mock_compute_angles, patched_load_single_traj, compute_angles_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    patched_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_angles
    expected_angles = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_compute_angles.return_value = expected_angles

    # Mock the path registry get_mapped_path method
    compute_angles_tool.path_registry.get_mapped_path = MagicMock(
        return_value="angles_plot.png"
    )

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    angle_indices = [(0, 1, 2), (1, 2, 3)]
    result = compute_angles_tool._run(traj_file, angle_indices, top_file)

    # Assertions
    patched_load_single_traj.assert_called_once_with(
        compute_angles_tool.path_registry, traj_file, top_file
    )
    mock_compute_angles.assert_called_once_with(
        mock_traj, angle_indices, periodic=True, opt=True
    )
    compute_angles_tool.path_registry.get_mapped_path.assert_called_once_with(
        "angles_plot.png"
    )
    mock_savefig.assert_called_once_with("angles_plot.png")
    assert result == "Succeeded. Bond angles computed, saved to file and plot saved."


def test_run_fail_compute_angles(patched_load_single_traj, compute_angles_tool):
    # Simulate the trajectory loading failure
    patched_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    angle_indices = [(0, 1, 2), (1, 2, 3)]
    result = compute_angles_tool._run(traj_file, angle_indices, top_file)

    # Assertions
    patched_load_single_traj.assert_called_once_with(
        compute_angles_tool.path_registry, traj_file, top_file
    )
    assert result == "Failed. Trajectory could not be loaded."


# Similar tests for other classes (ComputeChi1, ComputeChi2, etc.)


@patch("matplotlib.pyplot.savefig")
@patch("mdtraj.compute_phi")
@patch("mdtraj.compute_psi")
def test_run_success_ramachandran_plot(
    mock_compute_psi,
    mock_compute_phi,
    mock_savefig,
    patched_load_single_traj,
    ramachandran_plot,
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    patched_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_phi and compute_psi
    expected_phi = ([(0, 1, 2, 3)], [[0.7, 0.8, 0.9]])
    expected_psi = ([(0, 1, 2, 3)], [[1.0, 1.1, 1.2]])
    mock_compute_phi.return_value = expected_phi
    mock_compute_psi.return_value = expected_psi

    # Mock the path registry get_mapped_path method
    ramachandran_plot.path_registry.get_mapped_path = MagicMock(
        return_value="ramachandran_plot.png"
    )

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = ramachandran_plot._run(traj_file, top_file)

    # Assertions
    patched_load_single_traj.assert_called_once_with(
        ramachandran_plot.path_registry, traj_file, top_file
    )
    mock_compute_phi.assert_called_once_with(mock_traj, periodic=True, opt=True)
    mock_compute_psi.assert_called_once_with(mock_traj, periodic=True, opt=True)
    ramachandran_plot.path_registry.get_mapped_path.assert_called_once_with(
        "ramachandran_plot.png"
    )
    # Ensure savefig is called
    print(mock_savefig.call_args_list)
    mock_savefig.assert_called_once_with("ramachandran_plot.png")

    assert result == "Succeeded. Ramachandran plot generated and saved to file."
