from unittest.mock import patch

import numpy as np
import pytest

from mdagent.tools.base_tools.analysis_tools.inertia import (
    MomentOfInertia, calculate_moment_of_inertia, load_traj, save_to_csv,
)

# @pytest.fixture
# def moi_functions(get_registry):
#     registry = get_registry("raw", True)
#     traj_fileid = "rec0_butane_123456"
#     top_fileid = "top_sim0_butane_123456"
#     return MomentOfInertiaFunctions(registry, top_fileid, traj_fileid)

##### tests for  helper functions #####

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
    data = np.array([[1, 2], [3, 4]])
    with patch('os.path.exists', return_value=False):
        with patch('numpy.savetxt', return_value=None) as mock_savetxt:
            csv_path = save_to_csv(get_registry, data, 'test_id', 'Description of data')
            assert 'test_id.csv' in csv_path

### testing inertia code ###

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


# def test_compute_moi(moi_functions):
#     result_message = moi_functions.compute_moi()
#     assert "Average Moment of Inertia Tensor:" in result_message
#     assert "saved to:" in result_message

# def test_analyze_moi(moi_functions):
#     result_message = moi_functions.analyze_moi()
#     assert "Plot of principal moments of inertia over time saved as:" in result_message

