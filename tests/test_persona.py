"""Unit tests for the persona loader."""

import textwrap

import pytest

from catanatron.players.llm.persona import (
    ENV_OVERRIDE_DIR,
    Persona,
    PersonaNotFound,
    load_persona,
)


def test_load_default_persona():
    persona = load_persona("default")
    assert isinstance(persona, Persona)
    assert persona.name == "default"
    assert "Settlers of Catan" in persona.system_prompt
    assert persona.chat_instructions == ""


def test_load_friendly_persona():
    persona = load_persona("friendly")
    assert persona.name == "friendly"
    assert "cooperative" in persona.system_prompt.lower()
    assert persona.chat_instructions  # non-empty


def test_load_aggressive_persona():
    persona = load_persona("aggressive")
    assert persona.name == "aggressive"
    assert persona.chat_instructions  # non-empty


def test_persona_not_found_raises():
    with pytest.raises(PersonaNotFound):
        load_persona("does_not_exist_xyz")


def test_override_dir_takes_precedence(tmp_path, monkeypatch):
    override = tmp_path / "personas"
    override.mkdir()
    (override / "friendly.yaml").write_text(
        textwrap.dedent(
            """
            name: friendly
            system_prompt: "OVERRIDE SYSTEM"
            chat_instructions: "OVERRIDE CHAT"
            """
        ).strip()
    )
    monkeypatch.setenv(ENV_OVERRIDE_DIR, str(override))

    persona = load_persona("friendly")
    assert persona.system_prompt == "OVERRIDE SYSTEM"
    assert persona.chat_instructions == "OVERRIDE CHAT"


def test_override_dir_falls_back_to_bundled(tmp_path, monkeypatch):
    override = tmp_path / "personas"
    override.mkdir()  # empty override dir
    monkeypatch.setenv(ENV_OVERRIDE_DIR, str(override))

    persona = load_persona("default")
    assert "Settlers of Catan" in persona.system_prompt


def test_missing_fields_default_to_empty(tmp_path, monkeypatch):
    override = tmp_path / "personas"
    override.mkdir()
    (override / "bare.yaml").write_text("name: bare\n")
    monkeypatch.setenv(ENV_OVERRIDE_DIR, str(override))

    persona = load_persona("bare")
    assert persona.name == "bare"
    assert persona.system_prompt == ""
    assert persona.chat_instructions == ""
