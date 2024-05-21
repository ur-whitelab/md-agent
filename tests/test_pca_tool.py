from unittest.mock import patch

from mdagent.tools.base_tools import PCATool


def test_pca_tool(get_registry):
    reg = get_registry("raw", True, dynamic=True, include_hydrogens=True)
    pca_tool = PCATool(path_registry=reg)
    good_inputs = {
        "trajectory_fileid": "rec0_butane_123456",
        "topology_fileid": "top_sim0_butane_123456",
        "pc_percentage": "95",
        "analysis": "all",
        "selection": "all",
    }
    bad_input_files = {
        "trajectory_fileid": "rec0_butane_456456",
        "topology_fileid": "top_sim0_butane_456456",
        "pc_percentage": "Ninety Percent",
        "analysis": "all",
        "selection": "name CA",
    }

    error_catching = pca_tool._run(bad_input_files)
    assert "Trajectory File ID not in path registry" in error_catching
    assert "Topology File ID not in path registry" in error_catching
    assert "pc_percentage value must be a float" in error_catching
    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
        "matplotlib.pyplot.xlabel"
    ), patch("matplotlib.pyplot.ylabel"), patch("matplotlib.pyplot.title"), patch(
        "matplotlib.pyplot.savefig"
    ), patch(
        "matplotlib.pyplot.close"
    ), patch(
        "seaborn.PairGrid"
    ), patch(
        "seaborn.PairGrid.map"
    ):
        result = pca_tool._run(good_inputs)

    assert "Analyses done:" in result
    assert "Cosine Content of each PC: " in result

    # This two tests involve matplotlib, and are annoying to get done.
    # assert "PCA plots saved as" in result
    # assert "Scree Plot saved as" in result
