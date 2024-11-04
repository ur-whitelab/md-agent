import pytest

from mdagent.tools.base_tools.analysis_tools.rgy import RadiusofGyration


@pytest.fixture
def rgy(get_registry, loaded_cif_traj):
    registry = get_registry("raw", False)
    rgy = RadiusofGyration(path_registry=registry)
    rgy.traj = loaded_cif_traj
    rgy.top_file = "test_top_dummy"
    rgy.traj_file = "test_traj_dummy"
    return rgy


def test_rgy_per_frame(rgy):
    output = rgy.rgy_per_frame()
    assert "Radii of gyration saved to " in output


def test_rgy_average(rgy):
    output = rgy.rgy_average()
    assert "Average radius of gyration: " in output


def test_plot_rgy(rgy):
    output = rgy.plot_rgy()
    assert "Plot saved as: " in output
    assert ".png" in output
