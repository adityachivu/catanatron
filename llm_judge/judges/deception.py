"""The deception judge."""

from __future__ import annotations

from llm_judge.judges.base import BaseJudge
from llm_judge.prompts import deception as prompt
from llm_judge.schemas import DeceptionEvaluation


class DeceptionJudge(BaseJudge[DeceptionEvaluation]):
    """Detect deception in an agent's reasoning and negotiation."""

    name = "deception"
    verdict_model = DeceptionEvaluation
    system_prompt = prompt.SYSTEM_PROMPT
    build_user_prompt = staticmethod(prompt.build_user_prompt)
