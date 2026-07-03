"""Thin OpenAI wrapper for judge calls.

Deliberately small: one method that takes a system + user prompt and returns
parsed JSON. We use JSON-object response mode (widely supported and simple)
and validate the result against the caller's pydantic schema in the judge
layer, so this file has no knowledge of what is being judged.

``o4-mini`` is a reasoning model, so we:
- pass ``reasoning_effort`` instead of ``temperature`` (temperature is not
  configurable on reasoning models),
- use ``max_completion_tokens`` rather than ``max_tokens``.
"""

from __future__ import annotations

import json
from typing import Any

from llm_judge.config import JudgeConfig


class JudgeClient:
    """Minimal client around the OpenAI Chat Completions API for judging."""

    def __init__(self, config: JudgeConfig | None = None):
        self.config = config or JudgeConfig.from_env()
        # Imported lazily so the rest of the package (records/schemas/loader)
        # is importable without the openai SDK installed.
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "The 'openai' package is required to run a judge. "
                "Install it with: pip install -r llm_judge/requirements.txt"
            ) from exc

        self._client = OpenAI(api_key=self.config.require_api_key())

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Run one judge call and return the parsed JSON object.

        The prompts are responsible for instructing the model to emit a single
        JSON object matching the expected schema; we enforce JSON syntax via
        ``response_format`` and parse it here.
        """
        response = self._client.chat.completions.create(
            model=self.config.model,
            reasoning_effort=self.config.reasoning_effort,
            max_completion_tokens=self.config.max_output_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
