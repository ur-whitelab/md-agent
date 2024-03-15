import pytest
from openmm import unit
from openmm.app import PME, HBonds

from mdagent.tools.base_tools.simulation_tools import SetUpandRunFunction


@pytest.fixture
def setupandrun(get_registry):
    return SetUpandRunFunction(get_registry("raw", False))


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


def test_parse_temperature(setupandrun):
    result = setupandrun.parse_temperature("300k")
    expected_result = unit.Quantity(300, unit.kelvin)
    assert expected_result == result[0]


def parse_friction(setupandrun):
    result = setupandrun.parse_friction("1/ps")
    expected_result = unit.Quantity(1, unit.picoseconds)
    assert expected_result == result[0]


def test_parse_time(setupandrun):
    result = setupandrun.parse_timestep("1ns")
    expected_result = unit.Quantity(1, unit.nanoseconds)
    assert expected_result == result[0]


def test_parse_pressure(setupandrun):
    result = setupandrun.parse_pressure("1bar")
    expected_result = unit.Quantity(1, unit.bar)
    assert expected_result == result[0]


def test_process_parameters(setupandrun):
    parameters = {
        "nonbondedMethod": "PME",
        "constraints": "HBonds",
        "rigidWater": True,
    }
    result = setupandrun._process_parameters(parameters)
    expected_result = {
        "nonbondedMethod": PME,
        "constraints": HBonds,
        "rigidWater": True,
    }
    for key in expected_result:
        assert result[0][key] == expected_result[key]
