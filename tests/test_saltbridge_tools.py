from unittest.mock import Mock

from mdagent.tools.base_tools.analysis_tools.salt_bridge_tool import (
    SaltBridgeFunction,
    SaltBridgeTool,
)


def test_salt_bridge_tool_run(get_registry):
    registry = get_registry("raw", False)
    # Mocking the dependencies
    mock_salt_bridge_function = Mock(spec=SaltBridgeFunction)
    mock_salt_bridge_function.find_salt_bridges.return_value = (
        [(1, 2), (3, 4)],
        [5, 6],
        [("ARG", "ASP"), ("LYS", "GLU")],
    )

    salt_bridge_tool = SaltBridgeTool(registry)
    salt_bridge_tool.salt_bridge_function = mock_salt_bridge_function

    # Running the tool
    result = salt_bridge_tool._run(
        "traj_file", "top_file", 0.4, [("ARG", "ASP"), ("LYS", "GLU")]
    )

    # Asserting the result
    assert result == ([(1, 2), (3, 4)], [5, 6], [("ARG", "ASP"), ("LYS", "GLU")])

    # Verifying that the function was called with the correct arguments
    mock_salt_bridge_function.find_salt_bridges.assert_called_once_with(
        "traj_file", "top_file", 0.4, [("ARG", "ASP"), ("LYS", "GLU")]
    )
