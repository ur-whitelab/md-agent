import mdtraj as md
import pytest

from mdagent.tools.base_tools.analysis_tools.salt_bridge_tool import SaltBridgeFunction

# pdb with salt bridge residues (ARG, ASP, LYS, GLU)
pdb_data = """
HEADER    MOCK SALT BRIDGE EXAMPLE
ATOM      1  N   ARG A   1       0.000   0.000   0.000
ATOM      2  CA  ARG A   1       1.000   0.000   0.000
ATOM      3  C   ARG A   1       1.500   1.000   0.000
ATOM      4  O   ASP A   2       2.000   1.000   0.000
ATOM      5  N   LYS A   3       0.000   1.000   1.000
ATOM      6  CA  LYS A   3       1.000   1.000   1.000
ATOM      7  C   LYS A   3       1.500   2.000   1.000
ATOM      8  O   GLU A   4       2.000   2.000   1.000
ATOM      9  N   ASP A   2       3.000   1.000   0.000
ATOM     10  O   GLU A   4       4.000   2.000   1.000
ATOM     11  N   GLU A   4       2.000   2.000   0.000
ATOM     12  O   GLU A   4       4.000   2.000   1.000
ATOM     13  N   LYS A   3       0.0     3.0      0.000
ATOM     14  O   LYS A   3       0.0     4.0      0.000
END
"""


@pytest.fixture
def get_salt_bridge_function(get_registry):
    # Create the SaltBridgeFunction object using the PDB file path
    reg = get_registry("raw", True)
    pdb_path = f"{reg.ckpt_dir}/sb_residues.pdb"
    with open(pdb_path, "w") as file:
        file.write(pdb_data)
    fxn = SaltBridgeFunction(reg)
    fxn.traj = md.load(pdb_path)
    fxn.traj_file = "sb_residues"
    # fxn._load_traj(pdb_path, pdb_path)  # Using pdb_path as both traj and top file for simplicity
    return fxn


@pytest.fixture
def get_salt_bridge_function_with_butane(get_registry):
    registry = get_registry("raw", True)
    traj_fileid = "rec0_butane_123456"
    top_fileid = "top_sim0_butane_123456"
    fxn = SaltBridgeFunction(registry)
    fxn._load_traj(traj_fileid, top_fileid)
    return fxn


def test_find_salt_bridges_with_salt_bridges(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    salt_bridge_function.find_salt_bridges()
    assert len(salt_bridge_function.salt_bridge_counts) == 1
    assert len(salt_bridge_function.salt_bridge_pairs) == 1  # Only 1 frame
    assert len(salt_bridge_function.salt_bridge_pairs[0][1]) == 6
    assert salt_bridge_function.salt_bridge_counts == [6]


def test_salt_bridge_files_single_frame(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    salt_bridge_function.find_salt_bridges()
    file_id = salt_bridge_function.save_results_to_file()
    fig_id = salt_bridge_function.plot_salt_bridge_counts()
    assert file_id is not None
    assert fig_id is None


def test_salt_bridge_files_multiple_frames(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    n_frames = 5
    multi_frame_traj = md.join([salt_bridge_function.traj] * n_frames)
    salt_bridge_function.traj = multi_frame_traj
    salt_bridge_function.find_salt_bridges()
    file_id = salt_bridge_function.save_results_to_file()
    fig_id = salt_bridge_function.plot_salt_bridge_counts()
    assert file_id is not None
    assert fig_id is not None


def test_no_salt_bridges(get_salt_bridge_function_with_butane):
    salt_bridge_function = get_salt_bridge_function_with_butane
    salt_bridge_function.find_salt_bridges()
    file_id = salt_bridge_function.save_results_to_file()
    fig_id = salt_bridge_function.plot_salt_bridge_counts()
    assert file_id is None
    assert fig_id is None
    assert len(salt_bridge_function.salt_bridge_counts) == 0
    assert len(salt_bridge_function.salt_bridge_pairs) == 0
    assert salt_bridge_function.salt_bridge_pairs == []
    assert file_id is None
    assert fig_id is None


def test_invalid_trajectory(get_salt_bridge_function):
    salt_bridge_function = get_salt_bridge_function
    salt_bridge_function.traj = None
    with pytest.raises(Exception, match="MDTrajectory hasn't been loaded"):
        salt_bridge_function.find_salt_bridges()
