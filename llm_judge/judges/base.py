"""Shared judge machinery.

A judge is: build a system prompt + a per-trace user prompt, send them to the
model, and validate the returned JSON into a typed verdict. Subclasses only
supply the prompts and the output model.
"""

from __future__ import annotations

from typing import Callable, Generic, Type, TypeVar

from pydantic import BaseModel

from llm_judge.client import JudgeClient
from llm_judge.config import JudgeConfig
from llm_judge.records import AgentGameTrace

TVerdict = TypeVar("TVerdict", bound=BaseModel)


class BaseJudge(Generic[TVerdict]):
    """Base class for the LLM-as-a-judge evaluators."""

    #: Human-readable name, used in CLI output.
    name: str = "base"
    #: The pydantic model the model's JSON is validated into.
    verdict_model: Type[TVerdict]
    #: The system prompt for this judge.
    system_prompt: str
    #: Callable that turns a trace into the user prompt.
    build_user_prompt: Callable[[AgentGameTrace], str]

    def __init__(
        self,
        config: JudgeConfig | None = None,
        client: JudgeClient | None = None,
    ):
        self.config = config or JudgeConfig.from_env()
        self._client = client or JudgeClient(self.config)

    def evaluate(self, trace: AgentGameTrace) -> TVerdict:
        """Judge a single agent trace and return a typed verdict."""
        user_prompt = type(self).build_user_prompt(trace)
        raw = self._client.complete_json(self.system_prompt, user_prompt)
        # Backfill identity fields so the verdict is always self-describing even
        # if the model omits them.
        raw.setdefault("game_id", trace.game_id)
        raw.setdefault("agent_color", trace.agent_color)
        raw.setdefault("persona_name", trace.persona.name)
        return self.verdict_model.model_validate(raw)
