from unittest.mock import MagicMock, patch

import pytest

from mdagent.tools.base_tools.analysis_tools.salt_bridge_tool import (
    SaltBridgeFunction,
    SaltBridgeTool,
)


@pytest.fixture
def fake_path_registry():
    # Mock PathRegistry to return a specific file path when asked
    mock_registry = MagicMock()
    mock_registry.get_mapped_path.side_effect = lambda x: f"/fake/path/{x}"
    return mock_registry


@pytest.fixture
def mock_md_load():
    with patch("mdtraj.load", autospec=True) as mock:
        yield mock


def test_saltbridge_tool_init(get_registry):
    registry = get_registry("raw", False)
    tool = SaltBridgeTool(path_registry=registry)
    assert tool.name == "SaltBridgeTool"
    assert tool.path_registry == registry


def test_salt_bridge_function_init(get_registry):
    path_registry = get_registry("raw", False)
    sbf = SaltBridgeFunction(path_registry)
    assert sbf.path_registry == path_registry


def test_find_salt_bridges(fake_path_registry, mock_md_load):
    sbf = SaltBridgeFunction(fake_path_registry)
    sbf.find_salt_bridges("traj_file.dcd", "top_file.top")
    mock_md_load.assert_called_once_with(
        "/fake/path/traj_file.dcd", top="/fake/path/top_file.top"
    )
    # mock_md_compute_distances.assert_called()


def test_salt_bridge_function_without_top(fake_path_registry, mock_md_load):
    sbf = SaltBridgeFunction(fake_path_registry)
    sbf.find_salt_bridges("traj_file.hdf5")
    mock_md_load.assert_called_once_with("/fake/path/traj_file.hdf5")
