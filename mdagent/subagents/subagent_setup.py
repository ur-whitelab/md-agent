from typing import Optional
from mdagent.tools import PathRegistry
from .agents import (
    ActionAgent, 
    CodeCriticAgent, 
    ExplorerAgent, 
    RefiningCurriculumAgent, 
    SkillAgent, 
    TaskCriticAgent
)

class SubAgentSettings:
    def __init__(
        self,
        path_registry: Optional[PathRegistry],
        subagents_model="gpt-3.5",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
    ):
        self.path_registry = path_registry
        self.subagents_model=subagents_model
        self.temp=temp
        self.max_iterations=max_iterations
        self.api_key=api_key
        self.verbose=verbose
        self.ckpt_dir=ckpt_dir
        self.resume=resume

class SubAgentInitializer:
    def __init__(self, settings: SubAgentSettings):
        self.path_registry = settings.path_registry
        self.subagents_model = settings.subagents_model
        self.temp = settings.temp
        self.max_iterations = settings.max_iterations
        self.api_key = settings.api_key
        self.verbose = settings.verbose
        self.ckpt_dir = settings.ckpt_dir
        self.resume = settings.resume

    def create_action(self):
        return ActionAgent(
            path_registry=self.path_registry,
            model=self.subagents_model,
            temp=self.temp,
            max_iterations=self.max_iterations,
            api_key=self.api_key,
            verbose=self.verbose,
        )

    def create_code_critic(self):
        return CodeCriticAgent(
            model=self.subagents_model,
            temp=self.temp,
            max_iterations=self.max_iterations,
            api_key=self.api_key,
            verbose=self.verbose,
        )

    def create_explorer(self):
        return ExplorerAgent(
            path_registry=self.path_registry,
            model=self.subagents_model,
            temp=self.temp,
            max_iterations=self.max_iterations,
            api_key=self.api_key,
            verbose=self.verbose,
            ckpt_dir=self.ckpt_dir,
            resume=self.resume
        )

    def create_refining_curriculum(self):
        return RefiningCurriculumAgent(
            path_registry=self.path_registry,
            model=self.subagents_model,
            temp=self.temp,
            max_iterations=self.max_iterations,
            api_key=self.api_key,
            verbose=self.verbose,
            ckpt_dir=self.ckpt_dir,
            resume=self.resume
        )

    def create_skill(self):
        return SkillAgent(
            path_registry=self.path_registry,
            model=self.subagents_model,
            temp=self.temp,
            max_iterations=self.max_iterations,
            api_key=self.api_key,
            verbose=self.verbose,
            ckpt_dir=self.ckpt_dir,
            resume=self.resume
        )

    def create_task_critic(self):
        return TaskCriticAgent(
            path_registry=self.path_registry,
            model=self.subagents_model,
            temp=self.temp,
            max_iterations=self.max_iterations,
            api_key=self.api_key,
            verbose=self.verbose,
        )

    def create_iteration_agents(self):
        return {
            "action": self.create_action(),
            "code_critic": self.create_code_critic(),
            "refining_curriculum": self.create_refining_curriculum(),
            "skill": self.create_skill(),
            "task_critic": self.create_task_critic(),
        }