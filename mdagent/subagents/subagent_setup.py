import warnings
from typing import Optional

from mdagent.subagents.agents import (
    Action,
    Critic,
    ExplorerAgent,
    RefiningCurriculumAgent,
    SkillManager,
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
    ):
        self.path_registry = path_registry
        self.subagents_model = subagents_model
        self.temp = temp
        self.max_iterations = max_iterations
        self.api_key = api_key
        self.verbose = verbose
        self.ckpt_dir = ckpt_dir
        self.resume = resume


class SubAgentInitializer:
    def __init__(self, settings: Optional[SubAgentSettings]):
        if settings is None:
            raise ValueError("Settings cannot be None")
        if settings.path_registry is None:
            warnings.warn(
                "path_registry is None, some agents may fail to be created.",
                UserWarning,
            )
        self.path_registry = settings.path_registry
        self.subagents_model = settings.subagents_model
        self.temp = settings.temp
        self.max_iterations = settings.max_iterations
        self.api_key = settings.api_key
        self.verbose = settings.verbose
        self.ckpt_dir = settings.ckpt_dir
        self.resume = settings.resume

    def create_action(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp
        }
        # Update params with any overrides
        params.update(overrides)
        return Action(**params)

    def create_critic(self, **overrides):
        params = {
            "model": self.subagents_model,
            "temp": self.temp,
        }
        # Update params with any overrides
        params.update(overrides)
        return Critic(**params)

    def create_explorer_agent(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp,
            "max_iterations": self.max_iterations,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "ckpt_dir": self.ckpt_dir,
            "resume": self.resume,
        }
        # Update params with any overrides
        params.update(overrides)
        return ExplorerAgent(**params)

    def create_refining_curriculum_agent(self, **overrides):
        params = {
            "model": self.subagents_model,
            "temp": self.temp,
            "max_iterations": self.max_iterations,
            "api_key": self.api_key,
            "verbose": self.verbose,
            "ckpt_dir": self.ckpt_dir,
            "resume": self.resume,
        }
        # Update params with any overrides
        params.update(overrides)
        return RefiningCurriculumAgent(**params)

    def create_skill_manager(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp,
            "ckpt_dir": self.ckpt_dir,
            "resume": self.resume,
        }
        # Update params with any overrides
        params.update(overrides)
        return SkillManager(**params)

    def create_iteration_agents(self, **overrides):
        return {
            "action": self.create_action(**overrides),
            "critic": self.create_critic(**overrides),
            "refining_curriculum": self.create_refining_curriculum_agent(**overrides),
            "skill": self.create_skill_manager(**overrides),
        }
