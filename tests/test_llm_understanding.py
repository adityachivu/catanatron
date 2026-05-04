"""
LLM Understanding Harness
=========================
Tests the LLM agent's comprehension of game and board state through natural-language
questions rather than action selection.

Each `UnderstandingScenario` constructs a deterministic game state, formats it
using the same `StateFormatter` pipeline as the live game, then asks an open-ended
question.  The agent uses a custom per-scenario system prompt and returns a plain
string — no action indices, no tools.

Scenarios are organised into four categories:
  GAME_STATE  – pure game-logic reasoning (VP calc, dev cards, largest army)
  TRADE       – trade evaluation and defensive reasoning
  BOARD_STATE – spatial / topological reasoning (adjacency, ports, robber hex)
  JOINT       – combined game + board reasoning (robber targeting, road races)

Scenario files
--------------
  tests/scenarios/board_scenarios.py   – board/game-state scenarios
  tests/scenarios/persona_scenarios.py – persona-driven scenarios

Usage
-----
# Run all scenarios with real API calls (requires CATAN_LLM_MODEL env var):
    python tests/test_llm_understanding.py

# Smoke-test without API calls (uses TestModel):
    CATAN_LLM_TEST_MODE=1 pytest tests/test_llm_understanding.py -v

# Run against a specific model:
    CATAN_LLM_MODEL=anthropic:claude-sonnet-4-20250514 python tests/test_llm_understanding.py

# Run specific scenarios by name:
    python tests/test_llm_understanding.py victory_point_calculation largest_army_tracking

Adding new scenarios
--------------------
Add board/game scenarios to  tests/scenarios/board_scenarios.py
Add persona scenarios to     tests/scenarios/persona_scenarios.py

How prompts are constructed (mirrors BaseLLMPlayer._build_prompt in players/llm/base.py)
-----------------------------------------------------------------------------------------
run_scenario() builds the user message in three sections:

  Section 1 — Header  (turn / phase / color / trade context)
  Section 2 — STRUCTURED_STATE_JSON  (identical format to the live agent)
  Section 3 — QUESTION  (understanding-test-specific; no equivalent in live game)

The system prompt comes from `UnderstandingScenario.system_prompt`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHERE TO CUSTOMIZE PROMPTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A) To change the SYSTEM PROMPT for a single scenario:
       Edit the `system_prompt` field of the UnderstandingScenario instance.
       Look for the comment  # ◀ PROMPT: system_prompt  below each scenario.

  B) To change the QUESTION for a single scenario:
       Edit the `question` field of the UnderstandingScenario instance.
       Look for the comment  # ◀ PROMPT: question  below each scenario.

  C) To use the live game agent's system prompt:
       Use the `persona` field on UnderstandingScenario (e.g. persona="default").
       This calls load_persona() and uses persona.system_prompt automatically.

  D) To change HOW THE STATE IS PRESENTED (prompt format / section order):
       Edit the  # ── PROMPT CUSTOMIZATION ──  block inside run_scenario().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap — makes the file runnable both via pytest and directly:
#   pytest: adds project root + catanatron/ via pytest.ini pythonpath setting
#   script: neither is on sys.path by default, so we add them here
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CATANATRON_SRC = _PROJECT_ROOT / "catanatron"
for _p in [str(_PROJECT_ROOT), str(_CATANATRON_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from pydantic_ai import Agent

from catanatron.players.llm.models import ModelInput, create_model
from catanatron.players.llm.persona import load_persona
from catanatron.players.llm.state_formatter import StateFormatter

from tests.scenarios import UnderstandingScenario
from tests.scenarios.board_scenarios import SCENARIOS as _BOARD_SCENARIOS
from tests.scenarios.persona_scenarios import SCENARIOS as _PERSONA_SCENARIOS


# ---------------------------------------------------------------------------
# Combined scenario registry
# ---------------------------------------------------------------------------

CATEGORIES = ("GAME_STATE", "TRADE", "BOARD_STATE", "JOINT")

SCENARIOS: list[UnderstandingScenario] = \
    _PERSONA_SCENARIOS
    #_BOARD_SCENARIOS + \


# ---------------------------------------------------------------------------
# Named-persona registry (auto-detected: files matching name_[a-d]_description)
# ---------------------------------------------------------------------------

_PERSONAS_DIR = (
    Path(__file__).resolve().parent.parent
    / "catanatron" / "catanatron" / "players" / "llm" / "personas"
)
NAMED_PERSONAS: list[str] = sorted(
    f.stem
    for f in _PERSONAS_DIR.glob("*.yaml")
    if re.match(r"^[a-z]+_[a-d]_", f.stem)
)

# Cross-product: every scenario × every named persona
PERSONA_SCENARIO_COMBOS: list[tuple[UnderstandingScenario, str]] = [
    (s, p) for s in SCENARIOS for p in NAMED_PERSONAS
]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: UnderstandingScenario,
    model: ModelInput = None,
    persona_override: str | None = None,
) -> str:
    """
    Execute a single understanding scenario and return the agent's answer.

    The user message is built to match BaseLLMPlayer._build_prompt() in
    players/llm/base.py — same section markers, same state JSON, same header.
    Sections 3-5 of _build_prompt (playable actions, strategy hints, output
    requirements) are omitted because this is a free-text comprehension test.
    Instead, a QUESTION section is appended.

    No tools are registered — this tests pure reasoning from the state JSON.

    ─── WHERE TO CHANGE PROMPTS ────────────────────────────────────────────
    System prompt  → scenario.system_prompt  (per-scenario, see each SCENARIO_* block)
                     or set scenario.persona="<name>" to use a persona's system_prompt
    Question       → scenario.question       (per-scenario, see each SCENARIO_* block)
    State format   → the  # ── PROMPT CUSTOMIZATION ──  block below
    ────────────────────────────────────────────────────────────────────────
    """
    game = scenario.setup()
    state = game.state
    parts: list[str] = []

    # Resolve persona (persona_override → scenario.persona → None)
    effective_persona_name = persona_override or scenario.persona
    resolved_persona = load_persona(effective_persona_name) if effective_persona_name else None
    resolved_system_prompt = resolved_persona.system_prompt if resolved_persona else scenario.system_prompt

    # ── PROMPT CUSTOMIZATION: Section 1 — Header ────────────────────────────
    # Mirrors BaseLLMPlayer._build_prompt() section 1.
    # Edit here to add/remove header lines shown to the model before the JSON.
    parts.append("=== CATAN GAME STATE ===")
    parts.append(f"Turn: {state.num_turns}")
    parts.append(f"Phase: {state.current_prompt.value}")
    parts.append(f"You are: {scenario.perspective_color.value}")
    if state.is_initial_build_phase:
        parts.append(f"Initial build phase: {state.is_initial_build_phase}")

    # Trade context (mirrors _build_prompt — shown when a trade is active)
    if state.is_resolving_trade:
        offer = state.current_trade[:5]
        ask = state.current_trade[5:10]
        resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
        offer_str = ", ".join(f"{resource_names[i]}: {v}" for i, v in enumerate(offer) if v > 0)
        ask_str = ", ".join(f"{resource_names[i]}: {v}" for i, v in enumerate(ask) if v > 0)
        parts.append(f"Active trade - Offering: [{offer_str}], Asking: [{ask_str}]")

    # Memory tools section (mirrors _build_prompt — injected when persona has memory enabled)
    if resolved_persona and resolved_persona.memory is not None:
        cfg = resolved_persona.memory
        parts.append("")
        parts.append("=== MEMORY TOOLS ===")
        if cfg.prompt_hint:
            parts.append(cfg.prompt_hint)
        parts.append(
            f"Budget remaining this turn: {cfg.max_reads_per_turn} read_memory, "
            f"{cfg.max_writes_per_turn} write_memory."
        )
        parts.append("=== END_MEMORY_TOOLS ===")

    parts.append("")  # blank line separator

    # ── PROMPT CUSTOMIZATION: Section 2 — State JSON ────────────────────────
    # Mirrors BaseLLMPlayer._build_prompt() section 2 exactly.
    # StateFormatter.format_full_state() is the single source of truth;
    # edit that function (players/llm/state_formatter.py) to change what
    # fields are included — don't duplicate logic here.
    game_state = StateFormatter.format_full_state(game, scenario.perspective_color)
    game_state.get("my_state", {}).pop("color", None)  # redundant with header "You are:"
    game_state_json = json.dumps(game_state, indent=2, default=str)
    parts.append("=== STRUCTURED_STATE_JSON ===")
    parts.append(game_state_json)
    parts.append("=== END_STRUCTURED_STATE_JSON ===")
    parts.append("")  # blank line separator

    # ── PROMPT CUSTOMIZATION: Section 3 — Question ──────────────────────────
    # Understanding-test-specific; no equivalent in the live agent.
    # To change the question for a specific test, edit scenario.question.
    parts.append("=== QUESTION ===")
    parts.append(scenario.question)
    parts.append("=== END QUESTION ===")

    user_prompt = "\n".join(parts)

    # ── PROMPT CUSTOMIZATION: System prompt ─────────────────────────────────
    # Each scenario supplies its own system_prompt.
    # When scenario.persona is set, the persona's system_prompt is used instead.
    # To use the live game prompt, set scenario.persona to a persona name (e.g. "default").
    resolved_model = create_model(model)
    agent: Agent[None, str] = Agent(
        resolved_model,
        system_prompt=resolved_system_prompt,
        output_type=str,
    )

    result = agent.run_sync(user_prompt)
    return result.output


# ---------------------------------------------------------------------------
# Results file helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "understanding_results"


def _write_categorised_results(
    results: list[tuple[UnderstandingScenario, str, str]],
    model_label: str,
) -> Path:
    """
    Write all scenario results to a single timestamped file, organised by scenario.
    Each scenario block shows all persona responses beneath the question.
    ``results`` is a list of (scenario, persona_name, answer) 3-tuples.
    Returns the path written to.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"{timestamp}_understanding_results.txt"

    lines = [
        "=" * 72,
        "  LLM UNDERSTANDING EVALUATION RESULTS",
        "=" * 72,
        f"  Timestamp : {datetime.now().isoformat()}",
        f"  Model     : {model_label}",
        f"  Results   : {len(results)} ({len(results)} scenario×persona pairs)",
        "=" * 72,
        "",
    ]

    # Group by scenario name (preserving scenario list order)
    seen_names: list[str] = []
    by_scenario: dict[str, list[tuple[UnderstandingScenario, str, str]]] = {}
    for scenario, persona_name, answer in results:
        if scenario.name not in by_scenario:
            seen_names.append(scenario.name)
            by_scenario[scenario.name] = []
        by_scenario[scenario.name].append((scenario, persona_name, answer))

    for sname in seen_names:
        entries = by_scenario[sname]
        scenario = entries[0][0]  # use first entry for metadata

        lines.append("─" * 72)
        lines.append(f"  SCENARIO: {scenario.name}  [{scenario.category}]")
        lines.append(f"  Testing  : {scenario.description}")
        lines.append(f"  Perspective: {scenario.perspective_color.value}")
        lines.append("─" * 72)
        lines.append("")
        lines.append("  QUESTION:")
        for qline in scenario.question.splitlines():
            lines.append(f"    {qline}")
        lines.append("")
        if scenario.desired_answer:
            lines.append("  DESIRED ANSWER:")
            for dline in scenario.desired_answer.strip().splitlines():
                lines.append(f"    {dline}")
            lines.append("")

        for _, persona_name, answer in entries:
            lines.append(f"  ── PERSONA: {persona_name} ──")
            for aline in answer.strip().splitlines():
                lines.append(f"    {aline}")
            lines.append("")

        lines.append("  " + "·" * 68)
        lines.append("")

    lines.append("=" * 72)
    lines.append("  END OF RESULTS")
    lines.append("=" * 72)

    content = "\n".join(lines)
    filename.write_text(content, encoding="utf-8")
    return filename


def _write_single_result(
    scenario: UnderstandingScenario,
    answer: str,
    model_label: str,
    persona_name: str | None = None,
) -> Path:
    """
    Write a single scenario result (used by pytest individual tests).
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"{scenario.name}__{persona_name}" if persona_name else scenario.name
    filename = RESULTS_DIR / f"{timestamp}_{slug}.txt"
    parts = [
        f"Scenario : {scenario.name}",
        f"Persona  : {persona_name or '(scenario default)'}",
        f"Category : {scenario.category}",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Model    : {model_label}",
        f"Testing  : {scenario.description}",
        f"Perspective: {scenario.perspective_color.value}",
        "",
        "QUESTION:",
        scenario.question,
        "",
    ]
    if scenario.desired_answer:
        parts += ["DESIRED ANSWER:", scenario.desired_answer.strip(), ""]
    parts += ["ANSWER:", answer.strip(), ""]
    content = "\n".join(parts)
    filename.write_text(content, encoding="utf-8")
    return filename


# ---------------------------------------------------------------------------
# Pytest tests (smoke: response is non-empty string)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario,persona_name",
    PERSONA_SCENARIO_COMBOS,
    ids=[f"{s.name}__{p}" for s, p in PERSONA_SCENARIO_COMBOS],
)
def test_scenario(scenario: UnderstandingScenario, persona_name: str):
    """
    Smoke test: agent produces a non-empty answer for each (scenario, persona) pair.
    Run with CATAN_LLM_TEST_MODE=1 to use TestModel (no API calls).
    (Actual results generation should be run via python CLI, not pytest).
    """
    answer = run_scenario(scenario, persona_override=persona_name)
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0, "Expected a non-empty answer from the agent"
    print(f"\n\n{'─' * 60}")
    print(f"Scenario : {scenario.name} [{scenario.category}]  Persona: {persona_name}")
    print(f"\nANSWER:\n{answer.strip()}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# CLI runner — runs all scenarios and writes one categorised output file
# ---------------------------------------------------------------------------

def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


def main():
    """
    Run all (scenario × named-persona) combos with real API calls, print answers to
    stdout, and save all results to one file grouped by scenario in understanding_results/.

    Pass scenario names as command-line arguments to run only those specific scenarios
    (still cross-products against all named personas).
    """
    args = sys.argv[1:]

    scenarios_to_run = SCENARIOS
    if args:
        scenarios_to_run = [s for s in SCENARIOS if s.name in args]
        if not scenarios_to_run:
            print(f"Error: No scenarios match the provided arguments: {args}")
            print("Available scenarios:", [s.name for s in SCENARIOS])
            return

    combos_to_run: list[tuple[UnderstandingScenario, str]] = [
        (s, p) for s in scenarios_to_run for p in NAMED_PERSONAS
    ]

    model = os.environ.get("CATAN_LLM_MODEL") or None
    model_label = str(create_model(model))
    RESULTS_DIR.mkdir(exist_ok=True)

    print(_separator("═"))
    print("  LLM UNDERSTANDING HARNESS")
    print(_separator("═"))
    print(f"  Model    : {model_label}")
    print(f"  Scenarios: {len(scenarios_to_run)} × {len(NAMED_PERSONAS)} personas = {len(combos_to_run)} combos")
    print(f"  Personas : {', '.join(NAMED_PERSONAS)}")
    print(f"  Output   : {RESULTS_DIR}/")
    print(_separator("═"))

    all_results: list[tuple[UnderstandingScenario, str, str]] = []

    for i, (scenario, persona_name) in enumerate(combos_to_run, 1):
        print(f"\n[{i}/{len(combos_to_run)}]  [{scenario.category}] {scenario.name.upper()}  persona={persona_name}")
        print(_separator())
        print(f"  Testing : {scenario.description}")
        print(f"  Perspective: {scenario.perspective_color.value}")
        print(_separator("·"))
        print(f"  QUESTION:\n  {scenario.question}")
        if scenario.desired_answer:
            print(_separator("·"))
            print(f"  DESIRED ANSWER:\n  {scenario.desired_answer}")
        print(_separator("·"))
        print("  Running scenario...", end="", flush=True)

        try:
            answer = run_scenario(scenario, model=model, persona_override=persona_name)
            all_results.append((scenario, persona_name, answer))
            print(f"\r  ANSWER  (persona={persona_name}):")
            print()
            for line in answer.strip().splitlines():
                print(f"    {line}")
        except Exception as exc:
            error_msg = f"ERROR: {exc}"
            all_results.append((scenario, persona_name, error_msg))
            print(f"\r  {error_msg}")

        print(_separator())

    # Write all results grouped by scenario
    out_path = _write_categorised_results(all_results, model_label)
    print(f"\n{'═' * 72}")
    print(f"  All results saved to: {out_path}")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
