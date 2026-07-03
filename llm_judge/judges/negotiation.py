"""The combined negotiation judge (persona compliance + deception)."""

from __future__ import annotations

from llm_judge.judges.base import BaseJudge
from llm_judge.prompts import negotiation as prompt
from llm_judge.schemas import NegotiationEvaluation


class NegotiationJudge(BaseJudge[NegotiationEvaluation]):
    """Judge an agent's negotiation messages for persona compliance and deception."""

    name = "negotiation"
    verdict_model = NegotiationEvaluation
    system_prompt = prompt.SYSTEM_PROMPT
    build_user_prompt = staticmethod(prompt.build_user_prompt)
