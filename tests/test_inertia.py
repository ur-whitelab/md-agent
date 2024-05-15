import pytest
import numpy as np

from mdagent.tools.base_tools.analysis_tools.inertia import MomentOfInertia, MomentOfInertiaFunctions


@pytest.fixture
def moi_functions(get_registry):
    registry = get_registry('clean', True)
    return MomentOfInertiaFunctions(registry, 'top_sim0_butane_123456')

def test_calculate_center_of_mass(moi_functions):
    center_of_mass = moi_functions.calculate_center_of_mass()
    expected_output = np.array([[3.555, 4.614, 0.0]])
    np.testing.assert_array_almost_equal(center_of_mass, expected_output)

def test_calculate_moment_of_inertia(moi_functions):
    inertia_tensor = moi_functions.calculate_moment_of_inertia()
    expected_tensor = np.array([[1.5, -0.5, 0], [-0.5, 1.5, 0], [0, 0, 2]])  # Hypothetical example values
    np.testing.assert_array_almost_equal(inertia_tensor, expected_tensor)

def test_compute_moi(moi_functions):
    result_message = moi_functions.compute_moi()
    assert "Average Moment of Inertia Tensor:" in result_message
    assert "saved to:" in result_message

def test_analyze_moi(moi_functions):
    result_message = moi_functions.analyze_moi()
    assert "Plot of principal moments of inertia over time saved as:" in result_message

