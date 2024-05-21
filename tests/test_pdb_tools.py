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
    expected_output = "PDB file for C1=CC=CC=C1 successfully created and saved to "
    valid_pdb = molpdb.small_molecule_pdb(valid_smiles)
    assert expected_output in valid_pdb and "benzene.pdb" in valid_pdb
    assert os.path.exists(f"{molpdb.path_registry.ckpt_pdb}/benzene.pdb")

    # test with invalid SMILES string and invalid molecule name
    invalid_smiles = "C1=CC=CC=C1X"
    invalid_name = "NotAMolecule"
    expected_output = (
        "There was an error getting pdb. Please input a single molecule name."
    )
    assert expected_output in molpdb.small_molecule_pdb(invalid_smiles)
    assert expected_output in molpdb.small_molecule_pdb(invalid_name)

    # test with valid molecule name
    valid_name = "water"
    expected_output = (
        "Succeeded. PDB file for water successfully created and " "saved to water.pdb."
    )
    assert molpdb.small_molecule_pdb(valid_name) == expected_output
    assert os.path.exists(f"{molpdb.path_registry.ckpt_pdb}/water.pdb")


def test_packmol_pdb_download_only(packmol):
    packmol.path_registry._remove_path_from_json("water")
    packmol.path_registry._remove_path_from_json("benzene")
    small_molecules = ["water", "benzene"]
    packmol._get_sm_pdbs(small_molecules)
    assert os.path.exists(f"{packmol.path_registry.ckpt_pdb}/water.pdb")
    assert os.path.exists(f"{packmol.path_registry.ckpt_pdb}/benzene.pdb")


def test_packmol_download_only_once(packmol):
    packmol.path_registry._remove_path_from_json("water")
    small_molecules = ["water"]
    packmol._get_sm_pdbs(small_molecules)
    assert os.path.exists(f"{packmol.path_registry.ckpt_pdb}/water.pdb")
    os.path.getmtime(f"{packmol.path_registry.ckpt_pdb}/water.pdb")

    # Call the function again with the same molecule
    packmol._get_sm_pdbs(small_molecules)
    os.path.getmtime(f"{packmol.path_registry.ckpt_pdb}/water.pdb")
    print(f"{packmol.path_registry.ckpt_pdb}/{small_molecules[0]}.pdb")

    assert os.path.exists(f"{packmol.path_registry.ckpt_pdb}/{small_molecules[0]}.pdb")

    time_before = os.path.getmtime(
        f"{packmol.path_registry.ckpt_pdb}/{small_molecules[0]}.pdb"
    )
    time.sleep(3)
    packmol._get_sm_pdbs(small_molecules)
    print(f"{packmol.path_registry.ckpt_pdb}/{small_molecules[0]}.pdb")
    time_after = os.path.getmtime(
        f"{packmol.path_registry.ckpt_pdb}/{small_molecules[0]}.pdb"
    )
    assert time_before == time_after


def test_packmol_sm_download_called(packmol):
    packmol.path_registry.map_path(
        "1A3N_144150", f"{packmol.path_registry.ckpt_pdb}/1A3N_144150.pdb", "pdb"
    )
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

    here = os.path.exists(f"{packmol.path_registry.ckpt_pdb}/{small_molecule[0]}.pdb")
    os.path.exists(f"{packmol.path_registry.ckpt_pdb}/{small_molecule[0]}.pdb")
    assert here  # or maybe_here
    time_before = os.path.getmtime(
        f"{packmol.path_registry.ckpt_pdb}/{small_molecule[0]}.pdb"
    )
    time.sleep(3)
    packmol._get_sm_pdbs(small_molecule)
    time_after = os.path.getmtime(
        f"{packmol.path_registry.ckpt_pdb}/{small_molecule[0]}.pdb"
    )
    assert time_before == time_after
    os.remove(f"{packmol.path_registry.ckpt_pdb}/{small_molecule[0]}.pdb")
