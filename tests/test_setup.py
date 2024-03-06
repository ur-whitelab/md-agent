import pytest
from openmm import unit

from mdagent.tools.base_tools.simulation_tools import SetUpandRunFunction
from mdagent.utils import PathRegistry


@pytest.fixture
def get_registry():
    return PathRegistry()


@pytest.fixture
def setupandrun(get_registry):
    return SetUpandRunFunction(get_registry)


def test_parse_cutoff(setupandrun):
    cutoff = unit.Quantity(1.0, unit.nanometers)
    result = setupandrun._parse_cutoff(cutoff)
    assert cutoff == result

    cutoff = 3.0
    result = setupandrun._parse_cutoff(cutoff)
    expected_result = unit.Quantity(cutoff, unit.nanometers)
    assert expected_result == result

    cutoff = "2angstroms"
    result = setupandrun._parse_cutoff(cutoff)
    expected_result = unit.Quantity(2.0, unit.angstroms)
    assert expected_result == result


def test_parse_cutoff_unknown_unit(setupandrun):
    with pytest.raises(ValueError) as e:
        setupandrun._parse_cutoff("2pc")
        assert "Unknown unit" in str(e.value)
