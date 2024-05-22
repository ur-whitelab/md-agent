import os
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mdagent.tools.base_tools.analysis_tools.time_correlation import TimeCorrelation


@pytest.fixture
def time_corr_fxns(get_registry):
    registry = get_registry("raw", False)
    registry.map_path("valid_id", "fake_file.csv")
    tc = TimeCorrelation(registry, "valid_id", 0.1, "property")
    return tc


def test_time_correlation_initialization_with_invalid_id(get_registry):
    registry = get_registry("raw", False)
    with pytest.raises(ValueError):
        TimeCorrelation(registry, "invalid_id", 0.1, "property")


def test_calculate_time_correlation(time_corr_fxns):
    np.loadtxt = MagicMock(return_value=np.array([1, 2, 3, 4, 5]))
    message = time_corr_fxns.calculate_time_correlation()
    assert "saved to" in message
    assert os.path.exists(
        f"{time_corr_fxns.path_registry.ckpt_figures}/property_time_correlation.csv"
    )


def test_plot_time_correlation(time_corr_fxns):
    plt.savefig = MagicMock()
    np.loadtxt = MagicMock(return_value=np.array([1, 2, 3, 4, 5]))
    time_corr_fxns.calculate_time_correlation()
    message = time_corr_fxns.plot_time_correlation()
    assert "plot ID" in message
