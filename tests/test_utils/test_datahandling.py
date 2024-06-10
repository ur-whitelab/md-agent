from unittest.mock import patch

import numpy as np
import pytest

from mdagent.utils import load_single_traj, save_to_csv


# simple helper functions for tools
def test_load_traj_only_topology(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        traj = load_single_traj(registry, "top_sim0_butane_123456")
        mocked_get_mapped_path.assert_called_once()
        assert traj is not None


def test_load_traj_topology_and_traj(get_registry):
    registry = get_registry("raw", True)
    with patch.object(
        registry, "get_mapped_path", wraps=registry.get_mapped_path
    ) as mocked_get_mapped_path:
        traj = load_single_traj(
            registry, "top_sim0_butane_123456", "rec0_butane_123456"
        )
        assert mocked_get_mapped_path.call_count == 2
        assert traj is not None


def test_load_traj_fail_top_fileid(get_registry):
    registry = get_registry("raw", False)
    with pytest.raises(ValueError) as exc:
        load_single_traj(registry, "top_invalid")
    assert "Topology File ID 'top_invalid' not found" in str(exc.value)


def test_load_traj_fail_traj_fileid(get_registry):
    registry = get_registry("raw", True)
    with pytest.raises(ValueError) as exc:
        load_single_traj(registry, "top_sim0_butane_123456", "traj_invalid")
    assert "Trajectory File ID 'traj_invalid' not found" in str(exc.value)


def test_save_to_csv(get_registry):
    registry = get_registry("raw", False)
    data = np.array([[1, 2], [3, 4]])
    with patch("os.path.exists", return_value=False):
        with patch("numpy.savetxt", return_value=None):
            csv_path = save_to_csv(registry, data, "test_id", "Description of data")
            assert "test_id" in csv_path
