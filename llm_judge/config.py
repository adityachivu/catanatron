"""Configuration for the judge model.

Keeps the judge model (``o4-mini``) and its API key separate from whatever
model the game agents themselves run on. The API key is read from the
environment, falling back to a ``.env`` file in the repository root so the
judge picks up the same convention the rest of the project uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# repo root == two levels up from this file (llm_judge/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_MODEL = "o4-mini"


def _load_dotenv_value(key: str) -> str | None:
    """Minimal ``.env`` reader for a single key (no extra dependency).

    Only used as a fallback when the variable is not already in the
    environment. Ignores comments and blank lines; does not do interpolation.
    """
    if not ENV_FILE.is_file():
        return None
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _resolve(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or _load_dotenv_value(key) or default


@dataclass(frozen=True)
class JudgeConfig:
    """Settings for a judge run.

    Attributes:
        model: OpenAI model id used for judging. Defaults to ``o4-mini``.
        api_key: OpenAI API key. Read from ``OPENAI_API_KEY`` (env or .env).
        reasoning_effort: ``o4-mini`` is a reasoning model; this maps to the
            OpenAI ``reasoning_effort`` parameter ("low" | "medium" | "high").
        max_output_tokens: cap on completion tokens for a single verdict.
    """

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    reasoning_effort: str = "medium"
    max_output_tokens: int = 8000

    @classmethod
    def from_env(cls, **overrides: object) -> "JudgeConfig":
        """Build a config from environment/.env, applying any overrides."""
        base = dict(
            model=_resolve("CATAN_JUDGE_MODEL", DEFAULT_MODEL),
            api_key=_resolve("OPENAI_API_KEY"),
            reasoning_effort=_resolve("CATAN_JUDGE_REASONING_EFFORT", "medium"),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)  # type: ignore[arg-type]

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No OpenAI API key found. Set OPENAI_API_KEY in your environment "
                f"or add it to {ENV_FILE}."
            )
        return self.api_key
