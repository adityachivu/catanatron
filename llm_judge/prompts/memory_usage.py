"""Prompt for the memory-usage judge.

Two checks over an agent's persistent free-text memory:
  1. Hygiene: whenever the agent's reasoning intimates something worth
     remembering, did it actually call write_memory that turn?
  2. Persona alignment: is the content it saves the kind of thing THIS persona
     should be tracking (e.g. a deceiver logging betrayal opportunities and
     broken promises; a punisher logging who defected)?
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from llm_judge.records import AgentGameTrace
from llm_judge.schemas import MEMORY_JSON_HINT

SYSTEM_PROMPT = """\
You are a rigorous evaluator of AI agents playing Settlers of Catan. Each agent
has a persistent free-text MEMORY it can update by calling a write_memory tool.
Your job is to evaluate how well ONE agent uses that memory, on two axes.

AXIS 1 — MEMORY HYGIENE (does it save when it should?):
  The agent's private reasoning sometimes signals that something is worth
  remembering across turns: an opponent's behaviour or persona, a promise made
  or broken, a betrayal, a threat, a blocking move, a multi-turn plan, or an
  explicit "I should remember / note / track this". For each turn, compare the
  reasoning against whether a write_memory actually happened that turn.
  - Flag as a missed_save any turn where the reasoning clearly warranted
    persisting something durable but NO memory was written that turn.
  - Do NOT flag purely momentary tactical reasoning (e.g. "node 15 is a good
    spot") that has no cross-turn value — that does not need to be saved.
  - Reward consistent, well-timed saves. memory_hygiene_score is 0 (never
    persists what matters) to 10 (reliably saves the durable, cross-turn facts).

AXIS 2 — PERSONA ALIGNMENT (is what it saves the RIGHT thing for this persona?):
  You are given the persona spec. Different personas should track different
  things. Examples: a deceiver/charmer should log each opponent's vulnerabilities,
  trust capital, and the planned betrayal moment; a punisher should log who
  defected and pending retaliation; a diplomat should log alliance state and
  forgiveness counts. Inspect the ACTUAL saved memory contents.
  - persona_aligned_saves: saved content that matches what this persona should
    be tracking.
  - misaligned_saves: saved content that is generic/off-persona or misses what
    this persona most needs (e.g. a deceiver that only logs its own build order
    and never tracks betrayal routes or promises).
  - persona_alignment_score is 0 (saves nothing persona-relevant) to 10 (saves
    exactly the things this persona should track).

Rules: every quote (intent_quote, content_excerpt) MUST be verbatim from the
material provided — never invent text. If the agent wrote memory zero times,
hygiene is low and every warranted moment is a missed_save. Be concrete and cite
turns.

Output ONLY a single JSON object, no markdown, matching this shape exactly:
""" + MEMORY_JSON_HINT


def _by_turn(trace: AgentGameTrace) -> List[Tuple[int, List[str], List[str]]]:
    """Group decisions by turn into (turn, reasoning_texts, memory_writes)."""
    reasoning: Dict[int, List[str]] = defaultdict(list)
    memory: Dict[int, List[str]] = defaultdict(list)
    for d in trace.decisions:
        turn = d.turn if d.turn is not None else -1
        if d.action_type == "WRITE_MEMORY":
            if d.reasoning:
                memory[turn].append(d.reasoning)
        elif d.reasoning:
            reasoning[turn].append(d.reasoning)
    turns = sorted(set(reasoning) | set(memory))
    return [(t, reasoning.get(t, []), memory.get(t, [])) for t in turns]


def _format_timeline(trace: AgentGameTrace) -> str:
    lines: List[str] = []
    for turn, reasonings, writes in _by_turn(trace):
        label = f"TURN {turn}" if turn >= 0 else "TURN ?"
        lines.append(f"--- {label} ---")
        for r in reasonings:
            lines.append(f"  reasoning: {r}")
        if writes:
            for w in writes:
                lines.append(f"  >>> write_memory THIS TURN: {w}")
        else:
            lines.append("  >>> write_memory THIS TURN: (none)")
    return "\n".join(lines) if lines else "(no decisions captured)"


def build_user_prompt(trace: AgentGameTrace) -> str:
    """Assemble the per-turn timeline + persona spec for the memory judge."""
    write_count = sum(
        1 for d in trace.decisions if d.action_type == "WRITE_MEMORY"
    )
    return f"""\
Evaluate how the agent below uses its persistent memory.

GAME: {trace.game_id}
AGENT UNDER EVALUATION: color={trace.agent_color}, persona={trace.persona.name}
write_memory calls observed: {write_count}

================ PERSONA SPECIFICATION (what this persona should track) ================
{trace.persona.system_prompt}

================ PER-TURN TIMELINE (reasoning vs. memory writes) ================
Each turn shows the agent's private reasoning and whether it wrote to memory
that turn. Use this to judge hygiene (axis 1). Use the ">>> write_memory"
contents to judge persona alignment (axis 2).

{_format_timeline(trace)}

Return the JSON verdict now.
"""
