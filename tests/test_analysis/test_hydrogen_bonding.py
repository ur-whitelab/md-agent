from unittest.mock import MagicMock, patch

import pytest

# Assuming the BakerHubbard class and load_single_traj function are in the `my_module`
# module
from mdagent.tools.base_tools.analysis_tools.hydrogen_bonding_tools import (
    BakerHubbard,
    KabschSander,
    WernetNilsson,
)


@pytest.fixture
def baker_hubbard_tool(get_registry):
    path_registry = get_registry("raw", True)
    return BakerHubbard(path_registry)


@pytest.fixture
def kabsch_sander_tool(get_registry):
    path_registry = get_registry("raw", True)
    return KabschSander(path_registry)


@pytest.fixture
def wernet_nilsson_tool(get_registry):
    path_registry = get_registry("raw", True)
    return WernetNilsson(path_registry)


@patch(
    "mdagent.tools.base_tools.analysis_tools.hydrogen_bonding_tools.load_single_traj"
)
@patch("mdtraj.baker_hubbard")
def test_run_success_baker_hubbard(
    mock_baker_hubbard, mock_load_single_traj, baker_hubbard_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from baker_hubbard
    expected_hbonds = [(1, 2, 3), (4, 5, 6)]
    mock_baker_hubbard.return_value = expected_hbonds

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = baker_hubbard_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        baker_hubbard_tool.path_registry, top_file, traj_file
    )
    mock_baker_hubbard.assert_called_once_with(
        mock_traj, 0.1, exclude_water=True, periodic=True, sidechain_only=False
    )
    assert result == expected_hbonds


@patch(
    "mdagent.tools.base_tools.analysis_tools.hydrogen_bonding_tools.load_single_traj"
)
def test_run_fail_baker_hubbard(mock_load_single_traj, baker_hubbard_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = baker_hubbard_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        baker_hubbard_tool.path_registry, top_file, traj_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.hydrogen_bonding_tools.load_single_traj"
)
@patch("mdtraj.kabsch_sander")
def test_run_success_kabsch_sander(
    mock_kabsch_sander, mock_load_single_traj, kabsch_sander_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from kabsch_sander
    expected_hbonds = ([(0, 1), (2, 3)], [0.5, 0.7])
    mock_kabsch_sander.return_value = expected_hbonds

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = kabsch_sander_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        kabsch_sander_tool.path_registry, top_file, traj_file
    )
    mock_kabsch_sander.assert_called_once_with(mock_traj)
    assert result == expected_hbonds


@patch(
    "mdagent.tools.base_tools.analysis_tools.hydrogen_bonding_tools.load_single_traj"
)
def test_run_fail_kabsch_sander(mock_load_single_traj, kabsch_sander_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = kabsch_sander_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        kabsch_sander_tool.path_registry, top_file, traj_file
    )
    assert result == "Trajectory could not be loaded."


@patch(
    "mdagent.tools.base_tools.analysis_tools.hydrogen_bonding_tools.load_single_traj"
)
@patch("mdtraj.wernet_nilsson")
def test_run_success_wernet_nilsson(
    mock_wernet_nilsson, mock_load_single_traj, wernet_nilsson_tool
):
    # Create a mock trajectory
    mock_traj = MagicMock()
    mock_load_single_traj.return_value = mock_traj

    # Define the expected output from wernet_nilsson
    expected_hbonds = [(1, 2, 3), (4, 5, 6)]
    mock_wernet_nilsson.return_value = expected_hbonds

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = wernet_nilsson_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        wernet_nilsson_tool.path_registry, top_file, traj_file
    )
    mock_wernet_nilsson.assert_called_once_with(
        mock_traj, exclude_water=True, periodic=True, sidechain_only=False
    )
    assert result == expected_hbonds


@patch(
    "mdagent.tools.base_tools.analysis_tools.hydrogen_bonding_tools.load_single_traj"
)
def test_run_fail_wernet_nilsson(mock_load_single_traj, wernet_nilsson_tool):
    # Simulate the trajectory loading failure
    mock_load_single_traj.return_value = None

    # Call the _run method
    traj_file = "rec0_butane_123456"
    top_file = "top_sim0_butane_123456"
    result = wernet_nilsson_tool._run(traj_file, top_file)

    # Assertions
    mock_load_single_traj.assert_called_once_with(
        wernet_nilsson_tool.path_registry, top_file, traj_file
    )
    assert result == "Trajectory could not be loaded."
