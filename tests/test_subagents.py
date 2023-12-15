import pytest
from dotenv import load_dotenv

from mdagent.subagents.subagent_setup import SubAgentInitializer, SubAgentSettings


@pytest.fixture(scope="session", autouse=True)
def set_env():
    load_dotenv()


def test_subagent_setup():
    settings = SubAgentSettings(path_registry=None)
    initializer = SubAgentInitializer(settings)
    subagents = initializer.create_iteration_agents()
    action = subagents["action"]
    skill = subagents["skill"]
    critic = subagents["critic"]
    curriculum = initializer.create_curriculum()
    assert action is not None
    assert critic is not None
    assert curriculum is not None
    assert skill is not None
