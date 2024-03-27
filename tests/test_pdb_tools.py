import os
import time

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
    assert os.path.exists(f"{molpdb.path_registry.ckpt_files}/pdb/benzene.pdb")
    os.remove(f"{molpdb.path_registry.ckpt_files}/pdb/benzene.pdb")  # Clean up

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
        "PDB file for water successfully created and " "saved to water.pdb."
    )
    assert molpdb.small_molecule_pdb(valid_name) == expected_output
    assert os.path.exists(f"{molpdb.path_registry.ckpt_files}/pdb/water.pdb")
    os.remove(f"{molpdb.path_registry.ckpt_files}/pdb/water.pdb")


def test_packmol_pdb_download_only(packmol):
    packmol.path_registry._remove_path_from_json("water")
    packmol.path_registry._remove_path_from_json("benzene")
    small_molecules = ["water", "benzene"]
    packmol._get_sm_pdbs(small_molecules)
    assert os.path.exists(f"{packmol.path_registry.ckpt_files}/pdb/water.pdb")
    assert os.path.exists(f"{packmol.path_registry.ckpt_files}/pdb/benzene.pdb")
    os.remove(f"{packmol.path_registry.ckpt_files}/pdb/water.pdb")
    os.remove(f"{packmol.path_registry.ckpt_files}/pdb/benzene.pdb")


def test_packmol_download_only_once(packmol):
    packmol.path_registry._remove_path_from_json("water")
    small_molecules = ["water"]
    packmol._get_sm_pdbs(small_molecules)
    assert os.path.exists(f"{packmol.path_registry.ckpt_files}/pdb/water.pdb")
    os.path.getmtime(f"{packmol.path_registry.ckpt_files}/pdb/water.pdb")

    # Call the function again with the same molecule
    packmol._get_sm_pdbs(small_molecules)
    os.path.getmtime(f"{packmol.path_registry.ckpt_files}/pdb/water.pdb")

    assert os.path.exists(
        f"{packmol.path_registry.ckpt_files}/pdb/{small_molecules[0]}.pdb"
    )

    time_before = os.path.getmtime(
        f"{packmol.path_registry.ckpt_files}/pdb/{small_molecules[0]}.pdb"
    )
    time.sleep(3)
    packmol._get_sm_pdbs(small_molecules)
    time_after = os.path.getmtime(
        f"{packmol.path_registry.ckpt_files}/pdb/{small_molecules[0]}.pdb"
    )
    assert time_before == time_after
    os.remove(f"{packmol.path_registry.ckpt_files}/pdb/{small_molecules[0]}.pdb")
