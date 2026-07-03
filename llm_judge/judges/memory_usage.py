"""The memory-usage judge."""

from __future__ import annotations

from llm_judge.judges.base import BaseJudge
from llm_judge.prompts import memory_usage as prompt
from llm_judge.schemas import MemoryUsageEvaluation


class MemoryUsageJudge(BaseJudge[MemoryUsageEvaluation]):
    """Check that the agent saves what it should, and saves persona-relevant things."""

    name = "memory"
    verdict_model = MemoryUsageEvaluation
    system_prompt = prompt.SYSTEM_PROMPT
    build_user_prompt = staticmethod(prompt.build_user_prompt)
