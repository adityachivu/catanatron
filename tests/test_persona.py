"""Unit tests for the persona loader."""

import textwrap

import pytest

from catanatron.players.llm.persona import (
    ENV_OVERRIDE_DIR,
    MemoryConfig,
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


def test_default_persona_has_no_memory():
    persona = load_persona("default")
    assert persona.memory is None


def test_default_with_memory_persona_loads_memory_config():
    persona = load_persona("default_with_memory")
    assert isinstance(persona.memory, MemoryConfig)
    assert persona.memory.enabled is True
    assert persona.memory.max_reads_per_turn >= 1
    assert persona.memory.max_writes_per_turn >= 1
    assert persona.memory.prompt_hint  # non-empty


def test_memory_disabled_when_enabled_flag_missing(tmp_path, monkeypatch):
    override = tmp_path / "personas"
    override.mkdir()
    (override / "p.yaml").write_text(
        textwrap.dedent(
            """
            name: p
            memory:
              max_reads_per_turn: 5
            """
        ).strip()
    )
    monkeypatch.setenv(ENV_OVERRIDE_DIR, str(override))

    persona = load_persona("p")
    # enabled missing/false → memory becomes None so the consumer-side
    # `if persona.memory:` short-circuits cleanly
    assert persona.memory is None


def test_memory_block_must_be_a_mapping(tmp_path, monkeypatch):
    override = tmp_path / "personas"
    override.mkdir()
    (override / "p.yaml").write_text(
        textwrap.dedent(
            """
            name: p
            memory: "not a mapping"
            """
        ).strip()
    )
    monkeypatch.setenv(ENV_OVERRIDE_DIR, str(override))

    with pytest.raises(ValueError):
        load_persona("p")
