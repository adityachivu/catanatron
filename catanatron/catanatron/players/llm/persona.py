"""Persona loading for LLM players.

A persona is a named YAML file containing prompt text:

    name: friendly
    system_prompt: |
      You are a cooperative Settlers of Catan player...
    chat_instructions: |
      Keep negotiation messages warm and constructive...

Lookup order for `load_persona(name)`:
  1. ``$CATAN_PERSONAS_DIR/{name}.yaml`` if the env var is set
  2. Bundled ``catanatron/players/llm/personas/{name}.yaml``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


BUNDLED_PERSONAS_DIR = Path(__file__).parent / "personas"
ENV_OVERRIDE_DIR = "CATAN_PERSONAS_DIR"


class PersonaNotFound(FileNotFoundError):
    """Raised when a persona name cannot be resolved in any search path."""


@dataclass(frozen=True)
class Persona:
    name: str
    system_prompt: str
    chat_instructions: str


def _search_paths(name: str) -> List[Path]:
    paths: List[Path] = []
    override = os.environ.get(ENV_OVERRIDE_DIR)
    if override:
        paths.append(Path(override) / f"{name}.yaml")
    paths.append(BUNDLED_PERSONAS_DIR / f"{name}.yaml")
    return paths


def load_persona(name: str) -> Persona:
    """Load a persona by name.

    Raises ``PersonaNotFound`` if no YAML file matches in any search path.
    """
    searched: List[Path] = []
    for path in _search_paths(name):
        searched.append(path)
        if path.is_file():
            return _load_from_path(path, name)

    raise PersonaNotFound(
        f"Persona {name!r} not found. Searched: "
        + ", ".join(str(p) for p in searched)
    )


def _load_from_path(path: Path, expected_name: str) -> Persona:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Persona file {path} must be a YAML mapping")

    return Persona(
        name=data.get("name", expected_name),
        system_prompt=(data.get("system_prompt") or "").strip(),
        chat_instructions=(data.get("chat_instructions") or "").strip(),
    )
