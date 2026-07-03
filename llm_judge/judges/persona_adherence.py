"""The persona-adherence judge."""

from __future__ import annotations

from llm_judge.judges.base import BaseJudge
from llm_judge.prompts import persona_adherence as prompt
from llm_judge.schemas import PersonaAdherenceEvaluation


class PersonaAdherenceJudge(BaseJudge[PersonaAdherenceEvaluation]):
    """Score how faithfully an agent follows its persona prompt."""

    name = "persona"
    verdict_model = PersonaAdherenceEvaluation
    system_prompt = prompt.SYSTEM_PROMPT
    build_user_prompt = staticmethod(prompt.build_user_prompt)
