import os

import pytest

from mdagent.tools.base_tools.preprocess_tools.packing import (
    Molecule,
    PackmolBox,
    PackMolTool,
)
from mdagent.utils import PathRegistry


@pytest.fixture
def get_registry():
    return PathRegistry()


@pytest.fixture
def packmolbox(get_registry):
    return PackmolBox(get_registry)


@pytest.fixture
def packmoltool(get_registry):
    return PackMolTool(get_registry)


@pytest.fixture
def dummy_molecule():
    return Molecule(
        filename="water", file_id=100, number_of_molecules=2, instructions=None
    )


def test_add_molecule(packmolbox, dummy_molecule):
    initial_length = len(packmolbox.molecules)
    packmolbox.add_molecule(dummy_molecule)
    assert len(packmolbox.molecules) == initial_length + 1


def test_generate_input_header(packmolbox):
    packmolbox.generate_input_header()
    # assert file packmol.inp exists
    assert os.path.isfile("packmol.inp")
    os.remove("packmol.inp")


def test_generate_input(packmolbox, dummy_molecule):
    packmolbox.add_molecule(dummy_molecule)
    output = packmolbox.generate_input()
    assert "structure water" in output
    assert "number 2" in output
    assert "end structure" in output
