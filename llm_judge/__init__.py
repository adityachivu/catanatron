"""LLM-as-a-judge evaluation for the Catan persona agents.

Two independent judges, each with its own prompt and CLI command:

- ``deception``: detect deception in an agent's reasoning and negotiation.
- ``persona``:   score how faithfully an agent follows its persona prompt.

See ``llm_judge/README.md`` for the full picture. The judges consume a
normalized :class:`llm_judge.records.AgentGameTrace`; producing those traces
from real game logs is the (currently stubbed) job of ``llm_judge.loader``.
"""

from llm_judge.config import JudgeConfig
from llm_judge.records import AgentGameTrace
from llm_judge.judges.deception import DeceptionJudge
from llm_judge.judges.memory_usage import MemoryUsageJudge
from llm_judge.judges.negotiation import NegotiationJudge
from llm_judge.judges.persona_adherence import PersonaAdherenceJudge

__all__ = [
    "JudgeConfig",
    "AgentGameTrace",
    "DeceptionJudge",
    "MemoryUsageJudge",
    "NegotiationJudge",
    "PersonaAdherenceJudge",
]
