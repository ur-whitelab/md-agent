from .action import Action
from .code_critic import CodeCritic
from .curriculum import Explorer, RefiningCurriculum
from .skill import Skill
from .task_critic import TaskCritic


class SubAgents:
    def __init__(
        self,
        subagents_model="gpt-3.5",
        temp=0.1,
        max_iterations=40,
        api_key=None,
        verbose=True,
        ckpt_dir="ckpt",
        resume=False,
    ):
        self.action_agent = Action(
            model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )
        self.code_critic = CodeCritic(
            model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )
        self.refine_curriculum_agent = RefiningCurriculum(
            model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )
        self.skill_agent = Skill(
            model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )
        self.task_critic = TaskCritic(
            model=subagents_model,
            temp=temp,
            max_iterations=max_iterations,
            api_key=api_key,
            verbose=verbose,
        )


__all__ = [
    "Action",
    "CodeCritic",
    "Explorer",
    "RefiningCurriculum",
    "Skill",
    "TaskCritic",
    "SubAgents",
]
