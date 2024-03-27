from typing import Optional

from mdagent.subagents.agents import (
    Action,
    Critic,
    Curriculum,
    MemoryManager,
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
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
        retrieval_top_k=5,
        curriculum=True,
        memory: Optional[MemoryManager] = None,
        run_id="",
    ):
        self.path_registry = path_registry
        self.run_id = run_id
        self.memory = memory
        self.subagents_model = subagents_model
        self.temp = temp
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.ckpt_dir = ckpt_dir
        self.resume = resume
        self.retrieval_top_k = retrieval_top_k
        self.curriculum = curriculum


class SubAgentInitializer:
    def __init__(self, settings: Optional[SubAgentSettings] = None):
        if settings is None:
            settings = SubAgentSettings()
        if settings.path_registry is None:
            settings.path_registry = PathRegistry.get_instance()
        self.run_id = settings.run_id
        self.memory = settings.memory
        self.path_registry = settings.path_registry
        self.subagents_model = settings.subagents_model
        self.temp = settings.temp
        self.max_iterations = settings.max_iterations
        self.verbose = settings.verbose
        self.ckpt_dir = settings.ckpt_dir
        self.resume = settings.resume
        self.retrieval_top_k = settings.retrieval_top_k
        self.curriculum = settings.curriculum

    def create_action(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp,
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

    def create_curriculum(self, **overrides):
        if not self.curriculum:
            return None
        params = {
            "model": self.subagents_model,
            "temp": self.temp,
            "path_registry": self.path_registry,
        }
        # Update params with any overrides
        params.update(overrides)
        return Curriculum(**params)

    def create_skill_manager(self, **overrides):
        params = {
            "path_registry": self.path_registry,
            "model": self.subagents_model,
            "temp": self.temp,
            "ckpt_dir": self.ckpt_dir,
            "resume": self.resume,
            "retrieval_top_k": self.retrieval_top_k,
        }
        # Update params with any overrides
        params.update(overrides)
        return SkillManager(**params)

    def create_iteration_agents(self, **overrides):
        return {
            "action": self.create_action(**overrides),
            "critic": self.create_critic(**overrides),
            "skill": self.create_skill_manager(**overrides),
        }
