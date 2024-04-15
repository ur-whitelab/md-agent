from unittest.mock import Mock
import mdtraj as md 
import pytest

from mdagent.tools.base_tools.analysis_tools.salt_bridge_tool import SaltBridgeFunction
from mdagent.utils import PathRegistry

# define test data

traj_file = "test_trajectory.dcd"
top_file = "test_topology.pdb"

#Test saltbridgefunction class

def test_salt_bridge_function():
    #initalize SaltBridgeFunction
    salt_bridge_function = SaltBridgeFunction(path_registry=None)

#load traj using MDtraj 
    traj = md.load(traj_file, top=top_file)

#perform salt bridge analysis 

    salt_bridges, unpaired_residues, residue_pairs = salt_bridge_function.find.salt_bridges(traj, top=top_file)

#check to see if we get a list in our results

    assert isinstance(salt_bridges, list)

#Check to make sure residue pairs cant be changed ( tuple)

    for pair in residue_pairs:
         assert isinstance(pair, tuple) 

# check to make sure unpaired residue is assigned a number to identify it by the number (integer)
# to ensure the code works

    for residue in unpaired_residues:
        assert isinstance( residue, int)

# Finally, run the test
    if __name__ == "__main__":
        test_salt_bridge_function()
    


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
    traj = md.load(traj_file,top=top_file)  
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

