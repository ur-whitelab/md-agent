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
    action_agent = subagents["action"]
    skill_agent = subagents["skill"]
    critic = subagents["critic"]
    curriculum_agent = initializer.create_curriculum_agent()
    assert action_agent is not None
    assert critic is not None
    assert curriculum_agent is not None
    assert skill_agent is not None
