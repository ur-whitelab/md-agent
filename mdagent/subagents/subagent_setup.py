from typing import Optional

from mdagent.subagents.agents import (
    ActionAgent,
    CodeCriticAgent,
    CurriculumAgent,
    SkillManager,
    TaskCriticAgent,
)
from mdagent.utils import PathRegistry


class SubAgentSettings:
    def __init__(
        self,
        path_registry: Optional[PathRegistry] = None,
        subagents_model="gpt-3.5-turbo",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
        retrieval_top_k=5,
    ):
        self.path_registry = path_registry
        self.subagents_model = subagents_model
        self.temp = temp
        self.max_iterations = max_iterations
        self.api_key = api_key
        self.verbose = verbose
        self.ckpt_dir = ckpt_dir
        self.resume = resume
        self.retrieval_top_k = retrieval_top_k


class SubAgentInitializer:
    def __init__(self, settings: Optional[SubAgentSettings] = None):
        if settings is None:
            settings = SubAgentSettings()
        if settings.path_registry is None:
            # warnings.warn(
            #     "'path_registry' isn't specified. Use current directory by default.",
            #     UserWarning,
            # )
            settings.path_registry = PathRegistry.get_instance()
        self.path_registry = settings.path_registry
        self.subagents_model = settings.subagents_model
        self.temp = settings.temp
        self.max_iterations = settings.max_iterations
        self.api_key = settings.api_key
        self.verbose = settings.verbose
        self.ckpt_dir = settings.ckpt_dir
        self.resume = settings.resume
        self.retrieval_top_k = settings.retrieval_top_k

    def create_action_agent(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp,
            "max_iterations": self.max_iterations,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }
        # Update params with any overrides
        params.update(overrides)
        return ActionAgent(**params)

    def create_code_critic(self, **overrides):
        params = {
            "model": self.subagents_model,
            "temp": self.temp,
            "max_iterations": self.max_iterations,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }
        # Update params with any overrides
        params.update(overrides)
        return CodeCriticAgent(**params)

    def create_curriculum_agent(self, **overrides):
        params = {
            "model": self.subagents_model,
            "temp": self.temp,
            "verbose": self.verbose,
            "path_registry": self.path_registry,
        }
        # Update params with any overrides
        params.update(overrides)
        return CurriculumAgent(**params)

    def create_skill_manager(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp,
            "max_iterations": self.max_iterations,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "ckpt_dir": self.ckpt_dir,
            "resume": self.resume,
            "retrieval_top_k": self.retrieval_top_k,
        }
        # Update params with any overrides
        params.update(overrides)
        return SkillManager(**params)

    def create_task_critic(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp,
            "max_iterations": self.max_iterations,
            "api_key": self.api_key,
            "verbose": self.verbose,
        }
        # Update params with any overrides
        params.update(overrides)
        return TaskCriticAgent(**params)

    def create_iteration_agents(self, **overrides):
        return {
            "action": self.create_action_agent(**overrides),
            "code_critic": self.create_code_critic(**overrides),
            "skill": self.create_skill_manager(**overrides),
            "task_critic": self.create_task_critic(**overrides),
        }
