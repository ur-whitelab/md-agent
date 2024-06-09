from unittest.mock import MagicMock, patch

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
)


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


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_angles")
def test_run_success_compute_angles(
    mock_compute_angles, mock_load_single_traj, compute_angles_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_angles
    expected_angles = [0.1, 0.2, 0.3]
    mock_compute_angles.return_value = expected_angles

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    angle_indices = [(0, 1, 2), (1, 2, 3)]
    result = compute_angles_tool._run(traj_file, angle_indices, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_angles_tool.path_registry, traj_file, top_file
    )
    mock_compute_angles.assert_called_once_with(
        mock_traj, angle_indices, periodic=True, opt=True
    )
    assert result == expected_angles


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_angles(mock_load_single_traj, compute_angles_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    angle_indices = [(0, 1, 2), (1, 2, 3)]
    result = compute_angles_tool._run(traj_file, angle_indices, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_angles_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_dihedrals")
def test_run_success_compute_dihedrals(
    mock_compute_dihedrals, mock_load_single_traj, compute_dihedrals_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_dihedrals
    expected_dihedrals = [0.4, 0.5, 0.6]
    mock_compute_dihedrals.return_value = expected_dihedrals

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    indices = [(0, 1, 2, 3), (1, 2, 3, 4)]
    result = compute_dihedrals_tool._run(traj_file, indices, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_dihedrals_tool.path_registry, traj_file, top_file
    )
    mock_compute_dihedrals.assert_called_once_with(
        mock_traj, indices, periodic=True, opt=True
    )
    assert result == expected_dihedrals


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_dihedrals(mock_load_single_traj, compute_dihedrals_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    indices = [(0, 1, 2, 3), (1, 2, 3, 4)]
    result = compute_dihedrals_tool._run(traj_file, indices, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_dihedrals_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_phi")
def test_run_success_compute_phi(
    mock_compute_phi, mock_load_single_traj, compute_phi_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_phi
    expected_phi = [0.7, 0.8, 0.9]
    mock_compute_phi.return_value = expected_phi

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_phi_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_phi_tool.path_registry, traj_file, top_file
    )
    mock_compute_phi.assert_called_once_with(mock_traj, periodic=True, opt=True)
    assert result == expected_phi


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_phi(mock_load_single_traj, compute_phi_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_phi_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_phi_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_psi")
def test_run_success_compute_psi(
    mock_compute_psi, mock_load_single_traj, compute_psi_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_psi
    expected_psi = [1.0, 1.1, 1.2]
    mock_compute_psi.return_value = expected_psi

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_psi_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_psi_tool.path_registry, traj_file, top_file
    )
    mock_compute_psi.assert_called_once_with(mock_traj, periodic=True, opt=True)
    assert result == expected_psi


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_psi(mock_load_single_traj, compute_psi_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_psi_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_psi_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_chi1")
def test_run_success_compute_chi1(
    mock_compute_chi1, mock_load_single_traj, compute_chi1_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_chi1
    expected_chi1 = [1.3, 1.4, 1.5]
    mock_compute_chi1.return_value = expected_chi1

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi1_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi1_tool.path_registry, traj_file, top_file
    )
    mock_compute_chi1.assert_called_once_with(mock_traj, periodic=True, opt=True)
    assert result == expected_chi1


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_chi1(mock_load_single_traj, compute_chi1_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi1_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi1_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_chi2")
def test_run_success_compute_chi2(
    mock_compute_chi2, mock_load_single_traj, compute_chi2_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_chi2
    expected_chi2 = [1.6, 1.7, 1.8]
    mock_compute_chi2.return_value = expected_chi2

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi2_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi2_tool.path_registry, traj_file, top_file
    )
    mock_compute_chi2.assert_called_once_with(mock_traj, periodic=True, opt=True)
    assert result == expected_chi2


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_chi2(mock_load_single_traj, compute_chi2_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi2_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi2_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_chi3")
def test_run_success_compute_chi3(
    mock_compute_chi3, mock_load_single_traj, compute_chi3_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_chi3
    expected_chi3 = [1.9, 2.0, 2.1]
    mock_compute_chi3.return_value = expected_chi3

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi3_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi3_tool.path_registry, traj_file, top_file
    )
    mock_compute_chi3.assert_called_once_with(mock_traj, periodic=True, opt=True)
    assert result == expected_chi3


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_chi3(mock_load_single_traj, compute_chi3_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi3_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi3_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_chi4")
def test_run_success_compute_chi4(
    mock_compute_chi4, mock_load_single_traj, compute_chi4_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_chi4
    expected_chi4 = [2.2, 2.3, 2.4]
    mock_compute_chi4.return_value = expected_chi4

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi4_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi4_tool.path_registry, traj_file, top_file
    )
    mock_compute_chi4.assert_called_once_with(mock_traj, periodic=True, opt=True)
    assert result == expected_chi4


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_chi4(mock_load_single_traj, compute_chi4_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_chi4_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_chi4_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
@patch("mdtraj.compute_omega")
def test_run_success_compute_omega(
    mock_compute_omega, mock_load_single_traj, compute_omega_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from compute_omega
    expected_omega = [2.5, 2.6, 2.7]
    mock_compute_omega.return_value = expected_omega

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_omega_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_omega_tool.path_registry, traj_file, top_file
    )
    mock_compute_omega.assert_called_once_with(mock_traj, periodic=True, opt=True)
    assert result == expected_omega


@patch(
    "mdagent.tools.base_tools.analysis_tools.bond_angles_dihedrals_tool.load_single_traj"
)
def test_run_fail_compute_omega(mock_load_single_traj, compute_omega_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = compute_omega_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        compute_omega_tool.path_registry, traj_file, top_file
    )
    assert result == "Trajectory could not be loaded."
