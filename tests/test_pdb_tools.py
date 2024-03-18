import os
import time
from unittest.mock import MagicMock, patch

import pytest

from mdagent.tools.base_tools import get_pdb
from mdagent.tools.base_tools.preprocess_tools.packing import PackMolTool
from mdagent.tools.base_tools.preprocess_tools.pdb_get import MolPDB


@pytest.fixture
def fibronectin():
    return "fibronectin pdb"


@pytest.fixture
def molpdb(get_registry):
    return MolPDB(get_registry("raw", False))


@pytest.fixture
def packmol(get_registry):
    return PackMolTool(get_registry("raw", False))


def test_getpdb(fibronectin, get_registry):
    name, _ = get_pdb(fibronectin, get_registry("raw", False))
    assert name.endswith(".pdb")


def test_small_molecule_pdb(molpdb):
    # Test with a valid SMILES string
    valid_smiles = "C1=CC=CC=C1"  # Benzene
    expected_output = (
        "PDB file for C1=CC=CC=C1 successfully created and saved to "
        "files/pdb/benzene.pdb."
    )
    assert molpdb.small_molecule_pdb(valid_smiles) == expected_output
    assert os.path.exists("files/pdb/benzene.pdb")
    os.remove("files/pdb/benzene.pdb")  # Clean up

    # test with invalid SMILES string and invalid molecule name
    invalid_smiles = "C1=CC=CC=C1X"
    invalid_name = "NotAMolecule"
    expected_output = (
        "There was an error getting pdb. Please input a single molecule name."
    )
    assert molpdb.small_molecule_pdb(invalid_smiles) == expected_output
    assert molpdb.small_molecule_pdb(invalid_name) == expected_output

    # test with valid molecule name
    valid_name = "water"
    expected_output = (
        "PDB file for water successfully created and " "saved to files/pdb/water.pdb."
    )
    assert molpdb.small_molecule_pdb(valid_name) == expected_output
    assert os.path.exists("files/pdb/water.pdb")
    os.remove("files/pdb/water.pdb")  # Clean up


def test_packmol_sm_download_called(packmol):
    packmol.path_registry.map_path("1A3N_144150", "files/pdb/1A3N_144150.pdb", "pdb")
    with patch(
        "mdagent.tools.base_tools.preprocess_tools.packing.PackMolTool._get_sm_pdbs",
        new=MagicMock(),
    ) as mock_get_sm_pdbs:
        test_values = {
            "pdbfiles_id": ["1A3N_144150"],
            "small_molecules": ["water", "benzene"],
            "number_of_molecules": [1, 10, 10],
            "instructions": [
                ["inside box 0. 0. 0. 100. 100. 100."],
                ["inside box 0. 0. 0. 100. 100. 100."],
                ["inside box 0. 0. 0. 100. 100. 100."],
            ],
        }

        packmol._run(**test_values)

        mock_get_sm_pdbs.assert_called_with(["water", "benzene"])


@pytest.mark.parametrize("small_molecule", [["water"], ["benzene"]])
def test_packmol_download_only(packmol, small_molecule):
    packmol.path_registry._remove_path_from_json(f"{small_molecule[0]}")

    packmol._get_sm_pdbs(small_molecule)

    here = os.path.exists(f"files/pdb/{small_molecule[0]}.pdb")
    os.path.exists(f"tests/files/pdb/{small_molecule[0]}.pdb")
    assert here  # or maybe_here
    time_before = os.path.getmtime(f"files/pdb/{small_molecule[0]}.pdb")
    time.sleep(3)
    packmol._get_sm_pdbs(small_molecule)
    time_after = os.path.getmtime(f"files/pdb/{small_molecule[0]}.pdb")
    assert time_before == time_after
    os.remove(f"files/pdb/{small_molecule[0]}.pdb")
