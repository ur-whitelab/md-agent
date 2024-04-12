from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_traj():
    # Create a mock trajectory object
    traj = Mock()
    traj.topology.select.return_value = [1, 2, 3]  # Mocking residue selections
    return traj


def test_find_salt_bridges(mock_traj):
    # Instantiate SaltBridgeFunction
    salt_bridge_function = SaltBridgeFunction(path_registry=None)

    # Call find_salt_bridges method
    salt_bridges = salt_bridge_function.find_salt_bridges(traj=mock_traj)

    # Assert that salt_bridges contain the expected values
    expected_salt_bridges = [
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ]  # Example expected output
    assert salt_bridges == expected_salt_bridges


import pytest


def test_count_salt_bridges():
    salt_bridge_function = SaltBridgeFunction(path_registry=None)
    salt_bridges = [
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ]  # Example salt bridges
    count = salt_bridge_function.count_salt_bridges(salt_bridges)
    assert count == 6  # Example expected count
