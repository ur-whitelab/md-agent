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
    code_critic_agent = subagents["code_critic"]
    skill_agent = subagents["skill"]
    task_critic_agent = subagents["task_critic"]
    curriculum_agent = initializer.create_curriculum_agent()
    assert action_agent is not None
    assert code_critic_agent is not None
    assert curriculum_agent is not None
    assert skill_agent is not None
    assert task_critic_agent is not None
